"""Full training script"""
import logging
import random
import numpy as np
import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup, AdamW
import json
from model import RobertaClassifier, VADRobertaClassifier, EMDRobertaClassifier, AdapterSICLRobertaClassifier,\
    ConRobertaClassifier, AdapterRobertaClassifier, PredcitVADandClassfromLogit, AdapterSCCLClassifier
from utils import ErcTextDataset, get_num_classes, compute_metrics, set_seed, get_label_VAD, convert_label_to_VAD, compute_predicts
import os
import math
import argparse
import yaml
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support, precision_score, recall_score
import torch.cuda.amp.grad_scaler as grad_scaler
import torch.cuda.amp.autocast_mode as autocast_mode

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def train(epoch, model, optimizer, scheduler, loss_function, mode, data, batch_size, cuda, label_VAD, scaler):
    '''The training function for RobertaClassifier and AdapterRobertaClassifier.'''
    random.shuffle(data)
    if mode == 'train':
        model.train()
    else:
        model.eval()
    predicts = []
    ground_truth = []
    losses = []
    for i in range(0, len(data), batch_size):
        if mode == 'train':
            optimizer.zero_grad()
        bs_data = data[i: min(i+batch_size, len(data))]
        input_data = pad_sequence([torch.LongTensor(item['input_ids']) for item in bs_data], batch_first=True, padding_value=1)
        masks = pad_sequence([torch.LongTensor(item['attention_mask']) for item in bs_data], batch_first=True, padding_value=0)
        labels = torch.LongTensor([item['label'] for item in bs_data])
        #o_labels = [item['label'] for item in bs_data]
        #labels = convert_label_to_VAD(o_labels, label_VAD)
        if cuda:
            input_data = input_data.cuda()
            masks = masks.cuda()
            labels = labels.cuda()
        with autocast_mode.autocast():
            outputs = model(input_data, masks)
            #outputs, con_loss = model(input_data, masks, labels)
            loss = loss_function(outputs, labels)
            #loss = loss_function(outputs, labels) + 0.8*con_loss
        if mode == 'train':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
        ground_truth += labels.cpu().numpy().tolist()
        #ground_truth += o_labels
        predicts += torch.argmax(outputs, dim=1).cpu().numpy().tolist()
        #predicts += compute_predicts(outputs.cpu(), label_VAD)
        losses.append(loss.item())
    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(ground_truth, predicts) * 100, 2)
    weighted_f1 = round(f1_score(ground_truth, predicts, average='weighted') * 100, 2)
    micro_f1 = round(f1_score(ground_truth, predicts, average='micro') * 100, 2)
    macro_f1 = round(f1_score(ground_truth, predicts, average='macro') * 100, 2)
    if mode == 'train':
        print(
            "For epoch {}, train loss:{}, weighted F1 {}, micro F1 {}, macro F1 {}".format(epoch, avg_loss, weighted_f1,
                                                                                        micro_f1,
                                                                                        macro_f1))
    if mode == 'dev':
        print(
            "For epoch {}, dev loss:{}, weighted F1 {}, micro F1 {}, macro F1 {}".format(epoch, avg_loss, weighted_f1,
                                                                                        micro_f1,
                                                                                        macro_f1))
    if mode == 'test':
        print(
            "For epoch {}, test loss:{}, weighted F1 {}, micro F1 {}, macro F1 {}".format(epoch, avg_loss, weighted_f1,
                                                                                        micro_f1,
                                                                                        macro_f1))
        print(f1_score(ground_truth, predicts, average=None))


def train_vad(DATASET, epoch, model, optimizer, scheduler, loss_function, mode, data, batch_size, cuda, label_VAD, predicter, scaler):
    """The training function for AdapterSICLRobertaClassifier."""
    random.shuffle(data)
    #crossentropy_loss = nn.CrossEntropyLoss()
    if mode == 'train':
        model.train()
    else:
        model.eval()
    predicts = []
    ground_truth = []
    losses = []
    vads = []
    for i in range(0, len(data), batch_size):
        if mode == 'train':
            optimizer.zero_grad()
        bs_data = data[i: min(i+batch_size, len(data))]
        input_data = pad_sequence([torch.LongTensor(item['input_ids']) for item in bs_data], batch_first=True, padding_value=1)
        masks = pad_sequence([torch.LongTensor(item['attention_mask']) for item in bs_data], batch_first=True, padding_value=0)
        o_labels = [item['label'] for item in bs_data]
        labels = torch.LongTensor(o_labels)
        #one_hot_labels = nn.functional.one_hot(labels, num_classes=get_num_classes(DATASET))
        #one_hot_vad_labels = one_hot_labels.T
        vad_labels = convert_label_to_VAD(o_labels, label_VAD)
        #labels = [item['label'] for item in bs_data]
        if cuda:
            input_data = input_data.cuda()
            masks = masks.cuda()
            labels = labels.cuda()
            vad_labels = vad_labels.cuda()
        with autocast_mode.autocast():
            outputs, lda_loss, logits = model(input_data, masks, label_VAD, vad_labels)
            # emd_loss = torch.mean(loss_function(logits, labels))
            # vad_loss = loss_function(logits, vad_labels)
            ce_loss = loss_function(outputs, labels)
            loss = ce_loss + 0.8 * lda_loss
            # loss = vad_loss
        if mode == 'train':
            '''loss.backward()
            optimizer.step()
            scheduler.step()'''
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
        if mode == 'test':
            vads.append(logits.detach().cpu())
        ground_truth += labels.cpu().numpy().tolist()
        predicts += torch.argmax(outputs, dim=1).cpu().numpy().tolist()
        #predicts += predicter(logits, "cat").cpu().numpy().tolist()
        losses.append(loss.item())
    if DATASET is 'DailyDialog' and mode is not 'train':
        new_ground_truth = []
        new_predicts = []
        for gt, p in zip(ground_truth, predicts):
            if gt != 0:
                new_ground_truth.append(gt)
                new_predicts.append(p)
        ground_truth = new_ground_truth
        predicts = new_predicts
    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(ground_truth, predicts) * 100, 2)
    weighted_f1 = round(f1_score(ground_truth, predicts, average='weighted') * 100, 2)
    micro_f1 = round(f1_score(ground_truth, predicts, average='micro') * 100, 2)
    macro_f1 = round(f1_score(ground_truth, predicts, average='macro') * 100, 2)
    if mode == 'train':
        print(
            "For epoch {}, train loss:{}, weighted F1 {}, micro F1 {}, macro F1 {}".format(epoch, avg_loss, weighted_f1,
                                                                                        micro_f1,
                                                                                        macro_f1))
    if mode == 'dev':
        print(
            "For epoch {}, dev loss:{}, weighted F1 {}, micro F1 {}, macro F1 {}".format(epoch, avg_loss, weighted_f1,
                                                                                        micro_f1,
                                                                                        macro_f1))
    if mode == 'test':
        print(
            "For epoch {}, test loss:{}, weighted F1 {}, micro F1 {}, macro F1 {}".format(epoch, avg_loss, weighted_f1,
                                                                                        micro_f1,
                                                                                        macro_f1))
        vads = torch.cat(vads, dim=0)
        gt = torch.LongTensor(ground_truth)
        torch.save(vads, "./vads/vad_" + str(epoch) + ".pt")
        torch.save(gt, "./vads/label_" + str(epoch) + ".pt")
        print(f1_score(ground_truth, predicts, average=None))


def train_SCCL_vad(DATASET, epoch, model, optimizer, scheduler, loss_function, mode, data, batch_size, cuda, label_VAD, alpha, scaler):
    """The training function for AdapterSCCLClassifier."""
    random.shuffle(data)
    #crossentropy_loss = nn.CrossEntropyLoss()
    if mode == 'train':
        model.train()
    else:
        model.eval()
    predicts = []
    ground_truth = []
    losses = []
    num_classes = get_num_classes(DATASET)
    label_VAD = torch.stack(label_VAD, dim=0)
    vads = []
    #label_VAD = torch.rand(num_classes, 3)
    for i in range(0, len(data), batch_size):
        if mode == 'train':
            optimizer.zero_grad()
        bs_data = data[i: min(i+batch_size, len(data))]
        input_data = pad_sequence([torch.LongTensor(item['input_ids']) for item in bs_data], batch_first=True, padding_value=1)
        masks = pad_sequence([torch.LongTensor(item['attention_mask']) for item in bs_data], batch_first=True, padding_value=0)
        o_labels = [item['label'] for item in bs_data]
        labels = torch.LongTensor(o_labels)
        label_mask = torch.zeros(num_classes, dtype=torch.float)
        label_mask[o_labels] = 1.
        one_hot_labels = nn.functional.one_hot(labels, num_classes=num_classes)
        one_hot_vad_labels = one_hot_labels.T.float()
        if cuda:
            input_data = input_data.cuda()
            masks = masks.cuda()
            labels = labels.cuda()
            one_hot_vad_labels = one_hot_vad_labels.cuda()
            label_mask = label_mask.cuda()
            label_VAD = label_VAD.cuda()
        with autocast_mode.autocast():
            outputs, SCCL_loss, logits = model(input_data, masks, label_VAD, one_hot_vad_labels, label_mask)
            # emd_loss = torch.mean(loss_function(logits, labels))
            # vad_loss = loss_function(logits, vad_labels)
            ce_loss = loss_function(outputs, labels)
            loss = ce_loss + alpha * SCCL_loss
            # loss = vad_loss
        if mode == 'train':
            '''loss.backward()
            optimizer.step()
            scheduler.step()'''
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
        if mode == 'test':
            vads.append(logits.detach().cpu())
        ground_truth += labels.cpu().numpy().tolist()
        predicts += torch.argmax(outputs, dim=1).cpu().numpy().tolist()
        #predicts += predicter(logits, "cat").cpu().numpy().tolist()
        losses.append(loss.item())
    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(ground_truth, predicts) * 100, 2)
    weighted_f1 = round(f1_score(ground_truth, predicts, average='weighted') * 100, 2)
    if DATASET == 'DailyDialog':
        micro_f1 = round(f1_score(ground_truth, predicts, average='micro', labels=list(range(1, 7))) * 100, 2)
    else:
        micro_f1 = round(f1_score(ground_truth, predicts, average='micro') * 100, 2)
    macro_f1 = round(f1_score(ground_truth, predicts, average='macro') * 100, 2)
    if mode == 'train':
        print(
            "For epoch {}, train loss:{}, weighted F1 {}, micro F1 {}, macro F1 {}".format(epoch, avg_loss, weighted_f1,
                                                                                        micro_f1,
                                                                                        macro_f1))
    if mode == 'dev':
        print(
            "For epoch {}, dev loss:{}, weighted F1 {}, micro F1 {}, macro F1 {}".format(epoch, avg_loss, weighted_f1,
                                                                                        micro_f1,
                                                                                        macro_f1))
    if mode == 'test':
        print(
            "For epoch {}, test loss:{}, weighted F1 {}, micro F1 {}, macro F1 {}".format(epoch, avg_loss, weighted_f1,
                                                                                        micro_f1,
                                                                                        macro_f1))
        vads = torch.cat(vads, dim=0)
        gt = torch.LongTensor(ground_truth)
        torch.save(vads, "./vads/vad_"+str(epoch)+".pt")
        torch.save(gt, "./vads/label_"+str(epoch)+".pt")
        print(f1_score(ground_truth, predicts, average=None))


def main(CUDA: bool, LR: float, SEED: int, DATASET: str, BATCH_SIZE: int, model_checkpoint: str,
         speaker_mode: str, num_past_utterances: int, num_future_utterances: int,
         NUM_TRAIN_EPOCHS: int, WEIGHT_DECAY: float, WARMUP_RATIO: float, **kwargs):

    #ROOT_DIR = './multimodal-datasets/'
    ROOT_DIR = './data'
    NUM_CLASS = get_num_classes(DATASET)
    lr = float(LR)
    label_VAD = get_label_VAD(DATASET)

    '''Load data'''
    ds_train = ErcTextDataset(DATASET=DATASET, SPLIT='train', speaker_mode=speaker_mode,
                              num_past_utterances=num_past_utterances, num_future_utterances=num_future_utterances,
                              model_checkpoint=model_checkpoint,
                              ROOT_DIR=ROOT_DIR, SEED=SEED)

    ds_val = ErcTextDataset(DATASET=DATASET, SPLIT='val', speaker_mode=speaker_mode,
                            num_past_utterances=num_past_utterances, num_future_utterances=num_future_utterances,
                            model_checkpoint=model_checkpoint,
                            ROOT_DIR=ROOT_DIR, SEED=SEED)

    ds_test = ErcTextDataset(DATASET=DATASET, SPLIT='test', speaker_mode=speaker_mode,
                             num_past_utterances=num_past_utterances, num_future_utterances=num_future_utterances,
                             model_checkpoint=model_checkpoint,
                             ROOT_DIR=ROOT_DIR, SEED=SEED)
    tr_data = ds_train.inputs_
    dev_data = ds_val.inputs_
    test_data = ds_test.inputs_

    #model = VADRobertaClassifier(model_checkpoint, 768, NUM_CLASS)
    #model = RobertaClassifier(model_checkpoint, NUM_CLASS)
    #model = PrefixRobertaClassifier(model_checkpoint, 1024, NUM_CLASS)
    #model = AdapterRobertaClassifier(args, NUM_CLASS)
    #model = AdapterLDARobertaClassifier(args, NUM_CLASS)
    model = AdapterSCCLClassifier(args, NUM_CLASS)
    #model = ConRobertaClassifier(args, NUM_CLASS)
    #predicter = PredcitVADandClassfromLogit(args, label_type='single', label_VAD=label_VAD)
    predicter = None

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    '''Use linear scheduler.'''
    total_steps = float(10*len(ds_train.inputs_))/BATCH_SIZE
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps*WARMUP_RATIO), math.ceil(total_steps))
    loss_function = nn.CrossEntropyLoss()

    '''Due to the limitation of computational resources, we use mixed floating point precision.'''
    scaler = grad_scaler.GradScaler()
    #loss_function = nn.MSELoss()
    #loss_function = EMDLoss(args, label_type='single', label_VAD=label_VAD)

    if CUDA:
        model.cuda()
    #random.shuffle(tr_data)
    for n in range(NUM_TRAIN_EPOCHS):
        '''train(n, model, optimizer, scheduler, loss_function, "train", tr_data, BATCH_SIZE, CUDA, label_VAD, scaler)
        train(n, model, optimizer, scheduler, loss_function, "dev", dev_data, 2, CUDA, label_VAD, scaler)
        train(n, model, optimizer, scheduler, loss_function, "test", test_data, 2, CUDA, label_VAD, scaler)
        train_vad(DATASET, n, model, optimizer, scheduler, loss_function, "train", tr_data, BATCH_SIZE, CUDA, label_VAD, predicter, scaler)
        train_vad(DATASET, n, model, optimizer, scheduler, loss_function, "dev", dev_data, int(BATCH_SIZE/4), CUDA, label_VAD, predicter, scaler)
        train_vad(DATASET, n, model, optimizer, scheduler, loss_function, "test", test_data, int(BATCH_SIZE/4), CUDA, label_VAD, predicter, scaler)'''
        train_SCCL_vad(DATASET, n, model, optimizer, scheduler, loss_function, "train", tr_data, BATCH_SIZE, CUDA, label_VAD,
                  args['alpha'], scaler)
        train_SCCL_vad(DATASET, n, model, optimizer, scheduler, loss_function, "dev", dev_data, 1, CUDA,
                  label_VAD, args['alpha'], scaler)
        train_SCCL_vad(DATASET, n, model, optimizer, scheduler, loss_function, "test", test_data, 1, CUDA,
                  label_VAD, args['alpha'], scaler)
        print("-------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='erc RoBERTa text huggingface training')
    parser.add_argument('--DATASET', type=str, default="IEMOCAP")
    parser.add_argument('--CUDA', type=bool, default=True)
    parser.add_argument('--model_checkpoint', type=str, default="roberta-linadapter")
    parser.add_argument('--speaker_mode', type=str, default="upper")
    parser.add_argument('--num_past_utterances', type=int, default=1000)
    parser.add_argument('--num_future_utterances', type=int, default=1000)
    parser.add_argument('--BATCH_SIZE', type=int, default=4)
    parser.add_argument('--LR', type=float, default=1e-5)
    parser.add_argument('--HP_ONLY_UPTO', type=int, default=10)
    parser.add_argument('--NUM_TRAIN_EPOCHS', type=int, default=10)
    parser.add_argument('--WEIGHT_DECAY', type=float, default=0.01)
    parser.add_argument('--WARMUP_RATIO', type=float, default=0.2)
    parser.add_argument('--HP_N_TRIALS', type=int, default=5)
    parser.add_argument('--OUTPUT-DIR', type=str, default="./output")
    parser.add_argument('--SEED', type=int, default=42)
    parser.add_argument("--freeze_bert", default=False, type=bool,
                        help="freeze the parameters of original model.")
    parser.add_argument("--freeze_adapter", default=True, type=bool,
                        help="freeze the parameters of adapter.")
    parser.add_argument('--fusion_mode', type=str, default='add',
                        help='the fusion mode for bert feature and adapter feature |add|concat')

    parser.add_argument("--adapter_transformer_layers", default=2, type=int,
                        help="The transformer layers of adapter.")
    parser.add_argument("--adapter_size", default=768, type=int,
                        help="The hidden size of adapter.")
    parser.add_argument("--adapter_list", default="0,11,23", type=str,
                        help="The layer where add an adapter")
    parser.add_argument("--adapter_skip_layers", default=0, type=int,
                        help="The skip_layers of adapter according to bert layers")
    parser.add_argument('--meta_fac_adaptermodel', default="./pretrained_models/fac-adapter/pytorch_model.bin",
                        type=str, help='the pretrained factual adapter model')
    parser.add_argument('--meta_lin_adaptermodel', default="./pretrained_models/lin-adapter/pytorch_model.bin",
                        type=str, help='the pretrained linguistic adapter model')
    parser.add_argument('--alpha', default=0.8,
                        type=float, help='The loss coefficient.')


    args = parser.parse_args()
    args = vars(args)
    if "linadapter" in args['model_checkpoint']:
        args['meta_fac_adaptermodel'] = ''
    if "facadapter" in args['model_checkpoint']:
        args['meta_lin_adaptermodel'] = ''

    args['adapter_list'] = args['adapter_list'].split(',')
    args['adapter_list'] = [int(i) for i in args['adapter_list']]
    device = torch.device("cuda" if torch.cuda.is_available() and args['CUDA'] is True else "cpu")
    args['n_gpu'] = torch.cuda.device_count()
    args['device'] = device

    '''with open('./train-erc-text.yaml', 'r') as stream:
        args_ = yaml.load(stream, Loader=yaml.FullLoader)

    for key, val in args_.items():
        args[key] = val'''

    logging.info(f"arguments given to {__file__}: {args}")
    main(**args)