import torch
from torch import nn
import numpy as np
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, RobertaForMaskedLM, AutoModel, AutoTokenizer, AutoConfig
from modeling_roberta import PrefixRobertaModel
from pytorch_transformers.my_modeling_roberta import RobertaModelwithAdapter
from loss import EMDLoss, SupConLoss


class RobertaClassifier(nn.Module):
    """Fine-tune PLMs to directly predict categorical emotions."""
    def __init__(self, check_point, num_class):
        super(RobertaClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(check_point)
        self.tokenizer = AutoTokenizer.from_pretrained(check_point)
        self.config = AutoConfig.from_pretrained(check_point)
        hidden_size = self.config.hidden_size
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.dense2 = nn.Linear(hidden_size, num_class)

    def forward(self, x, mask):
        """
        :param x: The input of PLM. Dim: [B, seq_len, D]
        :param mask: The mask for input x. Dim: [B, seq_len]
        """
        x = self.bert(x, attention_mask=mask)[0]
        x = x[:, 0, :].squeeze(1)
        x = self.dropout(x)
        x = self.dense1(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.dense2(x)


class AdapterRobertaClassifier(nn.Module):
    """Fine-tune pre-trained knowledge adapter to predict categorical emotions."""
    def __init__(self, args, num_class):
        super(AdapterRobertaClassifier, self).__init__()
        self.bert = RobertaModelwithAdapter(args)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.config = RobertaConfig.from_pretrained('roberta-large')
        hidden_size = self.config.hidden_size
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.dense2 = nn.Linear(hidden_size, num_class)

    def forward(self, x, mask):
        x = self.bert(x, attention_mask=mask)[0]
        x = x[:, 0, :].squeeze(1)
        x = self.dropout(x)
        x = self.dense1(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.dense2(x)


class PrefixRobertaClassifier(nn.Module):
    """Fine-tune RoBERTa to predict categorical emotions with prefix-tuning."""
    def __init__(self, check_point, hidden_size, num_class):
        super(PrefixRobertaClassifier, self).__init__()
        self.bert = PrefixRobertaModel.from_pretrained(check_point)
        self.tokenizer = RobertaTokenizer.from_pretrained(check_point)
        self.config = RobertaConfig.from_pretrained(check_point)
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.dense2 = nn.Linear(hidden_size, num_class)
        prefix = self.tokenizer("The emotion is <mask>")['input_ids'][1: -1]
        prefix = self.bert.embeddings(torch.LongTensor(prefix).unsqueeze(0)).detach()
        self.prefix = nn.Parameter(prefix.clone(), requires_grad=True)

    def forward(self, x, mask):
        x, pfix = self.bert(x, prefix=self.prefix, attention_mask=mask)
        #x = x[:, 3, :].squeeze(1)
        x = pfix[:, 0, :].squeeze(1)
        x = self.dropout(x)
        x = self.dense1(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.dense2(x)


class VADRobertaClassifier(nn.Module):
    """Fine-tune RoBERTa to predict categorical emotions and VAD scores"""
    def __init__(self, check_point, hidden_size, num_class):
        super(VADRobertaClassifier, self).__init__()
        self.bert = RobertaForMaskedLM.from_pretrained(check_point)
        self.tokenizer = RobertaTokenizer.from_pretrained(check_point)
        self.config = RobertaConfig.from_pretrained(check_point)
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.dense21 = nn.Linear(hidden_size, num_class)
        self.dense22 = nn.Linear(hidden_size, 3)

    def forward(self, x, mask):
        x = self.bert(x, attention_mask=mask)[0]
        x = x[:, 0, :].squeeze(1)
        x = self.dropout(x)
        x = self.dense1(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.dense21(x), torch.sigmoid(self.dense22(x))


class AdapterSICLRobertaClassifier(nn.Module):
    """
    Supervised instance-level contrastive learning which directly contrast each instance with
    the corresponding emotion prototype.
    """
    def __init__(self, args, num_class):
        super(AdapterSICLRobertaClassifier, self).__init__()
        self.args = args
        if "adapter" in args['model_checkpoint']:
            self.bert = RobertaModelwithAdapter(args)
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
            self.config = RobertaConfig.from_pretrained('roberta-large')
        else:
            self.bert = RobertaModel.from_pretrained(args['model_checkpoint'])
            self.tokenizer = RobertaTokenizer.from_pretrained(args['model_checkpoint'])
            self.config = RobertaConfig.from_pretrained(args['model_checkpoint'])
        hidden_size = self.config.hidden_size
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.dense21 = nn.Linear(hidden_size, num_class)
        self.dense22 = nn.Linear(hidden_size, 3)
        self.MSE_loss = nn.MSELoss()

    def SICL_loss(self, x, vad_label, label_VAD, temperature=1):
        """
        :param x: The predicted VAD scores for each instance. Dim:[B, 1, 3]
        :param vad_label: The VAD label of each instance. Dim: [B, 3]
        :param label_VAD: The VAD prototype of each label. Dim: [label_num, 3]
        :param temperature: The temperature of contrastive learning.
        :return: The SICL loss.
        """
        '''Compute the distance between current instance and the corresponding label prototypes.'''
        logits = torch.div(torch.matmul(x, vad_label.unsqueeze(2)).squeeze(1), temperature) #[B, 1]

        '''Compute the distance between current instance and all label prototypes.'''
        if self.args["CUDA"]:
            label_VAD = torch.stack(label_VAD, dim=0).cuda() #[label_num, 3]
        else:
            label_VAD = torch.stack(label_VAD, dim=0) #[label_num, 3]
        label_VAD = label_VAD.unsqueeze(0).repeat(logits.shape[0], 1, 1).permute(0, 2, 1) #[B, 3, label_num]
        all_logits = torch.div(torch.matmul(x, label_VAD).squeeze(1), temperature) # [B, label_num]

        '''Compute contrastive loss.'''
        all_logits = torch.exp(all_logits)
        loss = logits-torch.log(all_logits.sum(1).unsqueeze(1))
        return -loss.mean()

    def mse_loss(self, x, vad_label):
        """Compute MSE loss between VADs."""
        return self.MSE_loss(x.squeeze(1), vad_label)

    def forward(self, x, mask, label_VAD, vad_labels):
        x = self.bert(x, attention_mask=mask)[0]
        x = x[:, 0, :].squeeze(1)
        x = self.dropout(x)
        x = self.dense1(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        vad = torch.sigmoid(self.dense22(x)).unsqueeze(1)
        loss = self.SICL_loss(vad, vad_labels, label_VAD)

        return self.dense21(x), loss, vad


class AdapterSCCLClassifier(nn.Module):
    """
    Supervised cluster-level contrastive learning which computes cluster-level VAD for each emotion and
    contrast with the emotion prototypes.
    """
    def __init__(self, args, num_class):
        super(AdapterSCCLClassifier, self).__init__()
        if "adapter" in args['model_checkpoint']:
            self.bert = RobertaModelwithAdapter(args)
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
            self.config = RobertaConfig.from_pretrained('roberta-large')
        else:
            self.bert = RobertaModel.from_pretrained(args['model_checkpoint'])
            self.tokenizer = RobertaTokenizer.from_pretrained(args['model_checkpoint'])
            self.config = RobertaConfig.from_pretrained(args['model_checkpoint'])
        hidden_size = self.config.hidden_size
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.dense21 = nn.Linear(hidden_size, num_class)
        self.dense22 = nn.Linear(hidden_size, 3)

    def SCCL_loss(self, x, label_VAD, one_hot_vad_labels, label_mask, temperature=1):
        """
        :param x: The predicted VAD scores. Dim:[B, 3]
        :param label_VAD: The VAD prototypes of each emotion. Dim:[label_num, 3]
        :param one_hot_vad_labels: One-hot matrix showing which instances has the emotion. Dim: [label_num, B]
        :param label_mask: One-hot vector that masks out the emotions that do not exist in current batch. Dim:[label_num]
        """
        '''Mask out unrelated instances for each emotion, and compute the cluster-level representation
         with the predicted VADs.'''
        masked_logits = one_hot_vad_labels.unsqueeze(2) * x.repeat(one_hot_vad_labels.shape[0], 1, 1) #[label_num, B, 3]
        logits = torch.mean(masked_logits, dim=1) #[label_num, 3]

        '''Compute logits for all clusters.'''
        logits = torch.div(torch.matmul(logits, label_VAD.T), temperature) #[label_num, label_num]

        '''Extract the logits to be maximised.'''
        up_logits = torch.diag(logits) #[label_num]

        '''Compute contrastive loss.'''
        all_logits = torch.log(torch.sum(torch.exp(logits), dim=1))
        loss = (up_logits-all_logits)*label_mask
        return -loss.mean()

    def mse_loss(self, x, vad_label):
        return self.MSE_loss(x.squeeze(1), vad_label)

    def forward(self, x, mask, label_VAD, one_hot_vad_labels, label_mask):
        x = self.bert(x, attention_mask=mask)[0]
        x = x[:, 0, :].squeeze(1)
        x = self.dropout(x)
        x = self.dense1(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        vad = torch.sigmoid(self.dense22(x))
        loss = self.SCCL_loss(vad, label_VAD, one_hot_vad_labels, label_mask)
        return self.dense21(x), loss, vad


class ConcatVADRobertaClassifier(nn.Module):
    """
    Concat the representation updated by categorical emotion detection and VAD prediction.
    """
    def __init__(self, check_point, hidden_size, num_class):
        super(ConcatVADRobertaClassifier, self).__init__()
        self.bert = RobertaModel.from_pretrained(check_point)
        self.tokenizer = RobertaTokenizer.from_pretrained(check_point)
        self.config = RobertaConfig.from_pretrained(check_point)
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.dense21 = nn.Linear(2*hidden_size, num_class)
        self.dense22 = nn.Linear(hidden_size, 3)

    def forward(self, x, mask):
        x = self.bert(x, attention_mask=mask)[0]
        x = x[:, 0, :].squeeze(1)
        x = self.dropout(x)
        x_m = self.dense1(x)
        x_a = self.dense2(x)
        x_m = torch.tanh(x_m)
        x_a = torch.tanh(x_a)
        x_m = self.dropout(x_m)
        x_a = self.dropout(x_a)
        return self.dense21(torch.cat([x_m, x_a], dim=-1)), torch.sigmoid(self.dense22(x_a))


class EMDRobertaClassifier(nn.Module):
    """
    The prediction model designed for EMD loss.
    """
    def __init__(self, check_point, label_VAD, num_class):
        super(EMDRobertaClassifier, self).__init__()
        self.bert = RobertaModel.from_pretrained(check_point)
        self.tokenizer = RobertaTokenizer.from_pretrained(check_point)
        self.config = RobertaConfig.from_pretrained(check_point)
        self.label_num = len(label_VAD)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.head = nn.Linear(self.config.hidden_size, self.label_num * 3)

        self.dense1 = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.dense2 = nn.Linear(self.config.hidden_size, num_class)

        '''self.category_label_vads = label_VAD
        v_scores = np.array([num[0] for num in self.category_label_vads])
        self.v_sorted_idxs = torch.tensor(np.argsort(v_scores).tolist()).cuda()
        a_scores = np.array([num[1] for num in self.category_label_vads])
        self.a_sorted_idxs = torch.tensor(np.argsort(a_scores).tolist()).cuda()
        d_scores = np.array([num[2] for num in self.category_label_vads])
        self.d_sorted_idxs = torch.tensor(np.argsort(d_scores).tolist()).cuda()
        self.v_sorted_values = torch.tensor(np.sort(v_scores).tolist()).cuda()
        self.a_sorted_values = torch.tensor(np.sort(a_scores).tolist()).cuda()
        self.d_sorted_values = torch.tensor(np.sort(d_scores).tolist()).cuda()

        self.v_head = nn.Linear(self.label_num, 1, bias=False)
        self.v_head.weight = nn.Parameter(torch.unsqueeze(self.v_sorted_values, 0))

        self.a_head = nn.Linear(self.label_num, 1, bias=False)
        self.a_head.weight = nn.Parameter(torch.unsqueeze(self.a_sorted_values, 0))

        self.d_head = nn.Linear(self.label_num, 1, bias=False)
        self.d_head.weight = nn.Parameter(torch.unsqueeze(self.d_sorted_values, 0))

        self.activation = nn.Sigmoid()'''


    def forward(self, x, mask):
        x = self.bert(x, attention_mask=mask)[0]
        x = x[:, 0, :].squeeze(1)

        x = self.dropout(x)
        x = self.dense1(x)
        logits = self.head(x)

        x = torch.tanh(x)
        x = self.dropout(x)

        '''v_logit, a_logit, d_logit = torch.split(logits, self.label_num, dim=1)  # logits for sorted (v, a, d)
        v_probs = self.activation(v_logit)
        a_probs = self.activation(a_logit)
        d_probs = self.activation(d_logit)
        v_logits = self.v_head(v_probs)
        a_logits = self.a_head(a_probs)
        d_logits = self.d_head(d_probs)
        logits = torch.cat((v_logits, a_logits, d_logits), dim=1)'''

        return x, logits


class PredcitVADandClassfromLogit(torch.nn.Module):

    def __init__(self, args, label_type, label_VAD):
        """
        The prediction model for EMD loss.
        """
        super(PredcitVADandClassfromLogit, self).__init__()
        self.args = args
        self.label_type = label_type
        self._check_args()
        self.category_label_vads = label_VAD

        if label_type == 'single':
            self.activation = nn.Softmax(dim=1)
            self.log_activation = nn.LogSoftmax(dim=1)
        else:  # 'multi'
            self.activation = nn.Sigmoid()
            self.log_activation = nn.LogSigmoid()

        self._sort_labels()

    def _check_args(self):
        assert self.label_type in ['single', 'multi']

    def _sort_labels(self):
        v_scores = [key[0] for key in self.category_label_vads]
        self.v_sorted_idxs = torch.tensor(np.argsort(v_scores).tolist())
        self.v_recover_idxs = torch.argsort(self.v_sorted_idxs)
        self.v_sorted_values = torch.tensor(np.sort(v_scores).tolist())

        a_scores = [key[1] for key in self.category_label_vads]
        self.a_sorted_idxs = torch.tensor(np.argsort(a_scores).tolist())
        self.a_recover_idxs = torch.argsort(self.a_sorted_idxs)
        self.a_sorted_values = torch.tensor(np.sort(a_scores).tolist())

        d_scores = [key[2] for key in self.category_label_vads]
        self.d_sorted_idxs = torch.tensor(np.argsort(d_scores).tolist())
        self.d_recover_idxs = torch.argsort(self.d_sorted_idxs)
        self.d_sorted_values = torch.tensor(np.sort(d_scores).tolist())

        if self.args['CUDA']:
            self.v_sorted_idxs = self.v_sorted_idxs.cuda()
            self.v_sorted_values = self.v_sorted_values.cuda()
            self.v_recover_idxs = self.v_recover_idxs.cuda()
            self.a_sorted_idxs = self.a_sorted_idxs.cuda()
            self.a_sorted_values = self.a_sorted_values.cuda()
            self.a_recover_idxs = self.a_recover_idxs.cuda()
            self.d_sorted_idxs = self.d_sorted_idxs.cuda()
            self.d_sorted_values = self.d_sorted_values.cuda()
            self.d_recover_idxs = self.d_recover_idxs.cuda()

    def forward(self, logits, predict):
        assert predict in ['vad', 'cat']
        """
        logits : (batch_size, 3*n_labels) # 3 for each (v, a, d)
        labels : (batch_size, n_labels) # only categorical labels
        """
        # 1. compute (sparse) p(v), p(a), p(d)
        v_logit, a_logit, d_logit = torch.split(logits, len(self.category_label_vads),
                                                dim=1)  # logits for sorted (v, a, d)
        v_probs = self.activation(v_logit)
        a_probs = self.activation(a_logit)
        d_probs = self.activation(d_logit)

        if predict == "vad":  # [ compute (v, a, d) == expected values ]
            e_v = torch.sum(v_probs * self.v_sorted_values, dim=1)
            e_a = torch.sum(a_probs * self.a_sorted_values, dim=1)
            e_d = torch.sum(d_probs * self.d_sorted_values, dim=1)
            predictions = torch.stack([e_v, e_a, e_d], dim=1)

        else:  # predict == 'cat': [ compute argmax(classes) ]
            v_logits_origin = torch.index_select(v_logit, 1, self.v_recover_idxs)
            a_logits_origin = torch.index_select(a_logit, 1, self.a_recover_idxs)
            d_logits_origin = torch.index_select(d_logit, 1, self.d_recover_idxs)
            class_logits_origin = v_logits_origin + a_logits_origin + d_logits_origin
            if self.label_type == 'multi':
                logprob = class_logits_origin - \
                          torch.log(torch.exp(v_logits_origin) + 1) - \
                          torch.log(torch.exp(a_logits_origin) + 1) - \
                          torch.log(torch.exp(d_logits_origin) + 1)
                predictions = torch.pow(torch.exp(logprob), 1 / 3) >= 0.5
                predictions = torch.squeeze(predictions.float())
            else:
                predictions = torch.max(class_logits_origin, 1)[1]  # argmax along dim=1

        return predictions


class ConRobertaClassifier(nn.Module):
    """Fine-tune RoBERTa model on categorical emotion detection and vanilla supervised contrastive learning."""
    def __init__(self, args, num_class):
        super(ConRobertaClassifier, self).__init__()
        self.bert = RobertaModel.from_pretrained(args['model_checkpoint'])
        self.tokenizer = RobertaTokenizer.from_pretrained(args['model_checkpoint'])
        self.config = RobertaConfig.from_pretrained(args['model_checkpoint'])
        hidden_size = self.config.hidden_size
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.dense2 = nn.Linear(hidden_size, num_class)
        self.dense21 = nn.Linear(hidden_size, 3)

    def forward(self, x, mask, labels):
        x = self.bert(x, attention_mask=mask)[0]
        x = x[:, 0, :].unsqueeze(1)
        x = self.dropout(x)
        x = self.dense1(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        vad = torch.sigmoid(self.dense21(x))
        con_rep = vad.clone().detach()
        con_loss = SupConLoss(features=torch.cat([vad, con_rep], dim=1), labels=labels)
        return self.dense2(x.squeeze(1)), con_loss