import torch
from torch import nn
import numpy as np


class EMDLoss(torch.nn.Module):

    def __init__(self, args, label_type, label_VAD):
        """
        The EMD loss is designed for the task type: "vad-from-categories"
        """
        super(EMDLoss, self).__init__()
        self.args = args
        self.label_type = label_type
        self._check_args()

        if label_type == 'single':
            self.activation = nn.Softmax(dim=1)
            self.ce_loss = torch.nn.CrossEntropyLoss()
        else:  # 'multi'
            self.activation = nn.Sigmoid()
            self.ce_loss = torch.nn.BCEWithLogitsLoss()

        self.category_label_vads = label_VAD
        self._sort_labels()

        self.eps = 1e-05

    def _check_args(self):
        assert self.label_type in ['single', 'multi']

    def _sort_labels(self):
        v_scores = [key[0] for key in self.category_label_vads]
        self.v_sorted_idxs = torch.tensor(np.argsort(v_scores).tolist())
        a_scores = [key[1] for key in self.category_label_vads]
        self.a_sorted_idxs = torch.tensor(np.argsort(a_scores).tolist())
        d_scores = [key[2] for key in self.category_label_vads]
        self.d_sorted_idxs = torch.tensor(np.argsort(d_scores).tolist())
        self.v_sorted_values = torch.tensor(np.sort(v_scores).tolist())
        self.a_sorted_values = torch.tensor(np.sort(a_scores).tolist())
        self.d_sorted_values = torch.tensor(np.sort(d_scores).tolist())
        if self.args['CUDA']:
            self.v_sorted_idxs = self.v_sorted_idxs.cuda()
            self.a_sorted_idxs = self.a_sorted_idxs.cuda()
            self.d_sorted_idxs = self.d_sorted_idxs.cuda()
            self.v_sorted_values = self.v_sorted_values.cuda()
            self.a_sorted_values = self.a_sorted_values.cuda()
            self.d_sorted_values = self.d_sorted_values.cuda()

    def _sort_labels_by_vad_coordinates(self, labels):
        v_labels = torch.index_select(labels, 1, self.v_sorted_idxs)
        a_labels = torch.index_select(labels, 1, self.a_sorted_idxs)
        d_labels = torch.index_select(labels, 1, self.d_sorted_idxs)
        return v_labels, a_labels, d_labels

    def _set_vad_distance_matrix(self):
        v_distance_vector = torch.roll(self.v_sorted_values, -1, 0) - self.v_sorted_values
        for idx, v_distance_element in enumerate(v_distance_vector):
            if v_distance_element == 0:
                assert idx != len(v_distance_vector) - 1
                v_distance_vector[idx] = v_distance_vector[idx + 1]
        v_distance_vector[-1] = 0
        a_distance_vector = torch.roll(self.a_sorted_values, -1, 0) - self.a_sorted_values
        for idx, a_distance_element in enumerate(a_distance_vector):
            if a_distance_element == 0:
                assert idx != len(a_distance_vector) - 1
                a_distance_vector[idx] = a_distance_vector[idx + 1]
        a_distance_vector[-1] = 0
        d_distance_vector = torch.roll(self.d_sorted_values, -1, 0) - self.d_sorted_values
        for idx, d_distance_element in enumerate(d_distance_vector):
            if d_distance_element == 0:
                assert idx != len(d_distance_vector) - 1
                d_distance_vector[idx] = d_distance_vector[idx + 1]
        d_distance_vector[-1] = 0
        return v_distance_vector, a_distance_vector, d_distance_vector

    def _intra_EMD_loss(self, input_probs, label_probs):
        intra_emd_loss = torch.div(torch.sum(
            torch.square(input_probs - label_probs), dim=1), len(self.category_label_vads))
        return intra_emd_loss

    def _inter_EMD_loss(self, input_probs, label_probs, distance):
        normalized_input_probs = input_probs / (torch.sum(input_probs, keepdim=True, dim=1) + self.eps)
        normalized_label_probs = label_probs / (torch.sum(label_probs, keepdim=True, dim=1) + self.eps)

        # multiply vad distance weight to subtraction of cumsum
        inter_emd_loss = torch.matmul(distance, torch.transpose(torch.square(
            torch.cumsum(normalized_input_probs, dim=1) - torch.cumsum(normalized_label_probs, dim=1),
        ), 0, 1))
        return inter_emd_loss

    def forward(self, logits, labels, use_emd=True):
        """
        logits : (batch_size, 3*n_labels) # 3 for each (v, a, d)
        labels : (batch_size, n_labels) # only categorical labels
        """

        if self.label_type == 'single':
            label_one_hot = torch.eye(len(self.category_label_vads))
            if self.args['CUDA']:
                label_one_hot = label_one_hot.cuda()
            labels = label_one_hot[labels]

        split_logits = torch.split(logits, len(self.category_label_vads), dim=1)  # logits for sorted (v, a, d)
        sorted_labels = self._sort_labels_by_vad_coordinates(labels)  # labels for sorted (v, a, d)
        distance_labels = self._set_vad_distance_matrix()

        if use_emd:
            losses = []
            for logit, sorted_label, distance_label in zip(split_logits, sorted_labels, distance_labels):
                input_probs = self.activation(logit)
                inter_emd_loss = self._inter_EMD_loss(input_probs, sorted_label, distance_label)
                intra_emd_loss = self._intra_EMD_loss(input_probs, sorted_label)
                emd_loss = inter_emd_loss + intra_emd_loss
                losses.append(emd_loss)
            loss = torch.mean(torch.stack(losses, dim=1), dim=1)

        else:  # using ce loss
            losses = torch.tensor(0.0).to(self.args.device)
            for logit, label in zip(split_logits, sorted_labels):
                if self.label_type == 'single':
                    label = torch.max(label, 1)[1]  # argmax along dim=1
                else:
                    label = label.type_as(logit)
                ce_loss = self.ce_loss(logit, label)
                losses += ce_loss
            loss = losses  # (sum of 3 dim)

        return loss


def SupConLoss(temperature=1., contrast_mode='all', features=None, labels=None, mask=None):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf. Multi-views are used to deal
    with cases when there is only one instance for certain emotion."""
    device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                         'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)  # 1 indicates two items belong to same class
    else:
        mask = mask.float().to(device)

    contrast_count = features.shape[1]  # num of views
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # (bsz * views, dim)
    if contrast_mode == 'one':
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature  # (bsz * views, dim)
        anchor_count = contrast_count  # num of views
    else:
        raise ValueError('Unknown mode: {}'.format(contrast_mode))

    '''compute logits'''
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T), temperature)  # (bsz, bsz)
    '''for numerical stability'''
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # (bsz, 1)
    logits = anchor_dot_contrast - logits_max.detach()  # (bsz, bsz) set max_value in logits to zero

    '''tile mask'''
    mask = mask.repeat(anchor_count, contrast_count)  # (anchor_cnt * bsz, contrast_cnt * bsz)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
                                0)  # (anchor_cnt * bsz, contrast_cnt * bsz)
    mask = mask * logits_mask  # 1 indicates two items belong to same class and mask-out itself

    '''compute log_prob'''
    exp_logits = torch.exp(logits) * logits_mask  # (anchor_cnt * bsz, contrast_cnt * bsz)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    '''compute mean of log-likelihood over positive'''
    if 0 in mask.sum(1):
        raise ValueError('Make sure there are at least two instances with the same class')
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

    # loss
    # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()
    return loss