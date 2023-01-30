"""utility and helper functions / classes."""
import torch
import json
import pickle
import os
import logging
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
import random
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def get_num_classes(DATASET: str) -> int:
    """Get the number of classes to be classified by dataset."""
    if DATASET == 'MELD':
        NUM_CLASSES = 7
    elif DATASET == 'IEMOCAP':
        NUM_CLASSES = 6
    elif DATASET == 'DailyDialog':
        NUM_CLASSES = 7
    elif DATASET == 'EmoryNLP':
        NUM_CLASSES = 7
    else:
        raise ValueError

    return NUM_CLASSES


def get_labels(DATASET: str):
    """Get labels of each dataset."""
    emotions = {}
    assert DATASET in ['MELD', 'IEMOCAP', 'DailyDialog', 'EmoryNLP']
    # MELD has 7 classes
    emotions['MELD'] = ['neutral',
                        'joy',
                        'surprise',
                        'anger',
                        'sadness',
                        'disgust',
                        'fear']

    # IEMOCAP originally has 11 classes but we'll only use 6 of them.
    emotions['IEMOCAP'] = ['neutral',
                           'frustration',
                           'sadness',
                           'anger',
                           'excited',
                           'happiness']

    emotions['DailyDialog'] = ['none',
                               'anger',
                               'disgust',
                               'fear',
                               'happiness',
                               'sadness',
                               'surprise']

    emotions['EmoryNLP'] = ['Joyful',
                            'Neutral',
                            'Powerful',
                            'Mad',
                            'Sad',
                            'Scared',
                            'Peaceful']

    return emotions[DATASET]


def get_label_VAD(DATASET: str):
    """Get VAD score of each label from the lexicon NRC-VAD."""
    emotions = get_labels(DATASET)
    label_VAD = []
    VADs = {}
    with open("./utils/NRC-VAD-Lexicon.txt", "r") as f:
        f.readline()
        for line in f:
            scores = line.split()
            key = scores[0] if len(scores) == 4 else " ".join(scores[:len(scores)-3])
            VADs[key] = torch.FloatTensor([float(scores[-3]), float(scores[-2]), float(scores[-1])])
            #VADs[key] = [float(scores[-3]), float(scores[-2]), float(scores[-1])]
    for i, emotion in enumerate(emotions):
        if emotion == 'none':
            label_VAD.append(VADs['Neutral'.lower()])
        else:
            label_VAD.append(VADs[emotion.lower()])
    return label_VAD


def convert_label_to_VAD(labels, labels_VAD):
    """Convert labels to corresponding VAD scores."""
    new_labels = []
    for label in labels:
        new_labels.append(labels_VAD[label])
    return torch.stack(new_labels, dim=0)


def compute_predicts(predicts, labels_VAD):
    """Predict categorical emotions from VADs."""
    f_p = []
    for i in range(predicts.shape[0]):
        f = []
        for vad in labels_VAD:
            f.append(float(torch.sum((predicts[i]-vad)**2)))
        f_p.append(f.index(min(f)))
    return f_p


def compute_metrics(eval_predictions) -> dict:
    """Return f1_weighted, f1_micro, and f1_macro scores."""
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)

    f1_weighted = f1_score(label_ids, preds, average='weighted')
    f1_micro = f1_score(label_ids, preds, average='micro')
    f1_macro = f1_score(label_ids, preds, average='macro')

    return {'f1_weighted': f1_weighted, 'f1_micro': f1_micro, 'f1_macro': f1_macro}


def set_seed(seed: int) -> None:
    """Set random seed to a fixed value.
       Set everything to be deterministic.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_emotion2id(DATASET: str) -> dict:
    """Get a dict that converts string class to numbers."""
    emotions = {}
    # MELD has 7 classes
    emotions['MELD'] = ['neutral',
                        'joy',
                        'surprise',
                        'anger',
                        'sadness',
                        'disgust',
                        'fear']

    # IEMOCAP originally has 11 classes but we'll only use 6 of them.
    emotions['IEMOCAP'] = ['neutral',
                           'frustration',
                           'sadness',
                           'anger',
                           'excited',
                           'happiness']

    emotions['DailyDialog'] = ['none',
                               'anger',
                               'disgust',
                               'fear',
                               'happiness',
                               'sadness',
                               'surprise']

    emotions['EmoryNLP'] = ['Joyful',
                            'Neutral',
                            'Powerful',
                            'Mad',
                            'Sad',
                            'Scared',
                            'Peaceful']

    emotion2id = {DATASET: {emotion: idx for idx, emotion in enumerate(
        emotions_)} for DATASET, emotions_ in emotions.items()}

    return emotion2id[DATASET]


class ErcTextDataset(torch.utils.data.Dataset):

    def __init__(self, DATASET='MELD', SPLIT='train', speaker_mode='upper',
                 num_past_utterances=0, num_future_utterances=0,
                 model_checkpoint='roberta-base',
                 ROOT_DIR='./data',
                 ONLY_UPTO=False, SEED=0):
        """Initialize emotion recognition in conversation text modality dataset class."""

        self.DATASET = DATASET
        self.ROOT_DIR = ROOT_DIR
        self.SPLIT = SPLIT
        self.speaker_mode = speaker_mode
        self.num_past_utterances = num_past_utterances
        self.num_future_utterances = num_future_utterances
        if "adapter" in model_checkpoint:
            self.model_checkpoint = 'roberta-large'
        else:
            self.model_checkpoint = model_checkpoint
        self.emotion2id = get_emotion2id(self.DATASET)
        self.ONLY_UPTO = ONLY_UPTO
        self.SEED = SEED
        if 'roberta' in model_checkpoint:
            self.sep_token = '</s></s>'
        else:
            self.sep_token = '[SEP]'

        if self.DATASET in ['MELD', 'IEMOCAP']:
            self._load_emotions()
            self._load_utterance_ordered()
            self._string2tokens()
            self.id2emotion = {val: key for key, val in self.emotion2id.items()}
            #print(self.utterance_ordered)
        else:
            set_seed(self.SEED)
            self._load_data()

    def _load_emotions(self):
        """Load the supervised labels"""
        with open(os.path.join(self.ROOT_DIR, self.DATASET, 'emotions.json'), 'r') as stream:
            self.emotions = json.load(stream)[self.SPLIT]

    def _dailydialog_load_utterance_speaker_emotion(self, utt, speaker_mode):
        """Load speaker information for EmoryNLP dataset."""
        utterance = utt['text'].strip()
        emotion = utt['label']
        if self.DATASET == 'DailyDialog':
            speaker = {'A': 'Mary',
                      'B': 'James'}
            if speaker_mode is not None and speaker_mode.lower() == 'upper':
                utterance = speaker[utt['speaker']].upper() + ': ' + utterance
            elif speaker_mode is not None and speaker_mode.lower() == 'title':
                utterance = speaker[utt['speaker']].title() + ': ' + utterance
        if self.DATASET == 'EmoryNLP':
            if speaker_mode is not None and speaker_mode.lower() == 'upper':
                utterance = utt['speaker'].upper() + ': ' + utterance
            elif speaker_mode is not None and speaker_mode.lower() == 'title':
                utterance = utt['speaker'].title() + ': ' + utterance
        return {'Utterance': utterance, 'Emotion': emotion}

    def _load_data(self):
        """Load data for EmoryNLP dataset."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_checkpoint, use_fast=True)
        max_model_input_size = tokenizer.max_model_input_sizes[self.model_checkpoint]
        num_truncated = 0
        inputs = []
        with open(os.path.join(self.ROOT_DIR, self.DATASET, self.SPLIT+'_data.json'), 'r') as f:
            raw_data = json.load(f)
            for dialogue in raw_data:
                ues = [self._dailydialog_load_utterance_speaker_emotion(utt, self.speaker_mode)
                           for utt in dialogue]
                num_tokens = [len(tokenizer(ue['Utterance'])['input_ids'])
                              for ue in ues]
                for idx, ue in enumerate(ues):
                    if ue['Emotion'] not in list(self.emotion2id.keys()):
                        continue

                    label = self.emotion2id[ue['Emotion']]

                    indexes = [idx]
                    indexes_past = [i for i in range(
                        idx - 1, idx - self.num_past_utterances - 1, -1)]
                    indexes_future = [i for i in range(
                        idx + 1, idx + self.num_future_utterances + 1, 1)]

                    offset = 0
                    if len(indexes_past) < len(indexes_future):
                        for _ in range(len(indexes_future) - len(indexes_past)):
                            indexes_past.append(None)
                    elif len(indexes_past) > len(indexes_future):
                        for _ in range(len(indexes_past) - len(indexes_future)):
                            indexes_future.append(None)

                    for i, j in zip(indexes_past, indexes_future):
                        if i is not None and i >= 0:
                            indexes.insert(0, i)
                            offset += 1
                            if sum([num_tokens[idx_] for idx_ in indexes]) > max_model_input_size:
                                del indexes[0]
                                offset -= 1
                                num_truncated += 1
                                break
                        if j is not None and j < len(ues):
                            indexes.append(j)
                            if sum([num_tokens[idx_] for idx_ in indexes]) > max_model_input_size:
                                del indexes[-1]
                                num_truncated += 1
                                break

                    utterances = [ues[idx_]['Utterance'] for idx_ in indexes]

                    if self.num_past_utterances == 0 and self.num_future_utterances == 0:
                        assert len(utterances) == 1
                        final_utterance = utterances[0]

                    elif self.num_past_utterances > 0 and self.num_future_utterances == 0:
                        if len(utterances) == 1:
                            final_utterance = self.sep_token + utterances[-1]
                        else:
                            final_utterance = ' '.join(
                                utterances[:-1]) + self.sep_token + utterances[-1]

                    elif self.num_past_utterances == 0 and self.num_future_utterances > 0:
                        if len(utterances) == 1:
                            final_utterance = utterances[0] + self.sep_token
                        else:
                            final_utterance = utterances[0] + \
                                              self.sep_token + ' '.join(utterances[1:])

                    elif self.num_past_utterances > 0 and self.num_future_utterances > 0:
                        if len(utterances) == 1:
                            final_utterance = self.sep_token + \
                                              utterances[0] + self.sep_token
                        else:
                            final_utterance = ' '.join(utterances[:offset]) + self.sep_token + \
                                              utterances[offset] + self.sep_token + \
                                              ' '.join(utterances[offset + 1:])
                    else:
                        raise ValueError

                    input_ids_attention_mask = tokenizer(final_utterance)
                    input_ids = input_ids_attention_mask['input_ids']
                    attention_mask = input_ids_attention_mask['attention_mask']

                    input_ = {'input_ids': input_ids,
                              'attention_mask': attention_mask, 'label': label}

                    inputs.append(input_)

            logging.info(f"number of truncated utterances: {num_truncated}")
            self.inputs_ = inputs

    def _load_utterance_ordered(self):
        """Load the ids of the utterances in order."""
        with open(os.path.join(self.ROOT_DIR, self.DATASET, 'utterance-ordered.json'), 'r') as stream:
            utterance_ordered = json.load(stream)[self.SPLIT]

        logging.debug(f"sanity check on if the text files exist ...")
        count = 0
        self.utterance_ordered = {}
        for diaid, uttids in utterance_ordered.items():
            self.utterance_ordered[diaid] = []
            for uttid in uttids:
                try:
                    with open(os.path.join(self.ROOT_DIR, self.DATASET, 'raw-texts', self.SPLIT, uttid + '.json'), 'r') as stream:
                        foo = json.load(stream)
                    self.utterance_ordered[diaid].append(uttid)
                except Exception as e:
                    count += 1
        if count != 0:
            logging.warning(f"number of not existing text files: {count}")
        else:
            logging.info(f"every text file exists fine!")

    def __len__(self):
        return len(self.inputs_)

    def _load_utterance_speaker_emotion(self, uttid, speaker_mode) -> dict:
        """Load an speaker-name prepended utterance and emotion label"""
        text_path = os.path.join(
            self.ROOT_DIR, self.DATASET, 'raw-texts', self.SPLIT, uttid + '.json')

        with open(text_path, 'r') as stream:
            text = json.load(stream)

        utterance = text['Utterance'].strip()
        emotion = text['Emotion']

        if self.DATASET == 'MELD':
            speaker = text['Speaker']
        elif self.DATASET == 'IEMOCAP':
            sessid = text['SessionID']
            # https: // www.ssa.gov/oact/babynames/decades/century.html
            speaker = {'Ses01': {'Female': 'Mary', 'Male': 'James'},
                       'Ses02': {'Female': 'Patricia', 'Male': 'John'},
                       'Ses03': {'Female': 'Jennifer', 'Male': 'Robert'},
                       'Ses04': {'Female': 'Linda', 'Male': 'Michael'},
                       'Ses05': {'Female': 'Elizabeth', 'Male': 'William'}}[sessid][text['Speaker']]

        else:
            raise ValueError(f"{self.DATASET} not supported!!!!!!")

        if speaker_mode is not None and speaker_mode.lower() == 'upper':
            utterance = speaker.upper() + ': ' + utterance
        elif speaker_mode is not None and speaker_mode.lower() == 'title':
            utterance = speaker.title() + ': ' + utterance

        return {'Utterance': utterance, 'Emotion': emotion}

    def _create_input(self, diaids, speaker_mode, num_past_utterances, num_future_utterances):
        """Create an input which will be an input to RoBERTa."""

        args = {'diaids': diaids,
                'speaker_mode': speaker_mode,
                'num_past_utterances': num_past_utterances,
                'num_future_utterances': num_future_utterances}

        logging.debug(f"arguments given: {args}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_checkpoint, use_fast=True)
        max_model_input_size = tokenizer.max_model_input_sizes[self.model_checkpoint]
        #max_model_input_size = tokenizer.max_model_input_sizes[self.model_checkpoint] - 4
        num_truncated = 0

        inputs = []
        for diaid in tqdm(diaids):
            ues = [self._load_utterance_speaker_emotion(uttid, speaker_mode)
                   for uttid in self.utterance_ordered[diaid]]

            num_tokens = [len(tokenizer(ue['Utterance'])['input_ids'])
                          for ue in ues]

            for idx, ue in enumerate(ues):
                if ue['Emotion'] not in list(self.emotion2id.keys()):
                    continue

                label = self.emotion2id[ue['Emotion']]

                indexes = [idx]
                indexes_past = [i for i in range(
                    idx-1, idx-num_past_utterances-1, -1)]
                indexes_future = [i for i in range(
                    idx+1, idx+num_future_utterances+1, 1)]

                offset = 0
                if len(indexes_past) < len(indexes_future):
                    for _ in range(len(indexes_future)-len(indexes_past)):
                        indexes_past.append(None)
                elif len(indexes_past) > len(indexes_future):
                    for _ in range(len(indexes_past) - len(indexes_future)):
                        indexes_future.append(None)

                for i, j in zip(indexes_past, indexes_future):
                    if i is not None and i >= 0:
                        indexes.insert(0, i)
                        offset += 1
                        if sum([num_tokens[idx_] for idx_ in indexes]) > max_model_input_size:
                            del indexes[0]
                            offset -= 1
                            num_truncated += 1
                            break
                    if j is not None and j < len(ues):
                        indexes.append(j)
                        if sum([num_tokens[idx_] for idx_ in indexes]) > max_model_input_size:
                            del indexes[-1]
                            num_truncated += 1
                            break

                utterances = [ues[idx_]['Utterance'] for idx_ in indexes]

                if num_past_utterances == 0 and num_future_utterances == 0:
                    assert len(utterances) == 1
                    final_utterance = utterances[0]

                elif num_past_utterances > 0 and num_future_utterances == 0:
                    if len(utterances) == 1:
                        final_utterance = self.sep_token + utterances[-1]
                    else:
                        final_utterance = ' '.join(
                            utterances[:-1]) + self.sep_token + utterances[-1]

                elif num_past_utterances == 0 and num_future_utterances > 0:
                    if len(utterances) == 1:
                        final_utterance = utterances[0] + self.sep_token
                    else:
                        final_utterance = utterances[0] + \
                            self.sep_token + ' '.join(utterances[1:])

                elif num_past_utterances > 0 and num_future_utterances > 0:
                    if len(utterances) == 1:
                        final_utterance = self.sep_token + \
                            utterances[0] + self.sep_token
                    else:
                        final_utterance = ' '.join(utterances[:offset]) + self.sep_token + \
                            utterances[offset] + self.sep_token + \
                            ' '.join(utterances[offset+1:])
                else:
                    raise ValueError
                input_ids_attention_mask = tokenizer(final_utterance)
                input_ids = input_ids_attention_mask['input_ids']
                attention_mask = input_ids_attention_mask['attention_mask']

                input_ = {'input_ids': input_ids,
                          'attention_mask': attention_mask, 'label': label}

                inputs.append(input_)

        logging.info(f"number of truncated utterances: {num_truncated}")
        return inputs

    def _string2tokens(self):
        """Convert string to (BPE) tokens."""
        logging.info(f"converting utterances into tokens ...")

        diaids = sorted(list(self.utterance_ordered.keys()))

        set_seed(self.SEED)
        random.shuffle(diaids)

        if self.ONLY_UPTO:
            logging.info(
                f"Using only the first {self.ONLY_UPTO} dialogues ...")
            diaids = diaids[:self.ONLY_UPTO]

        logging.info(f"creating input utterance data ... ")
        self.inputs_ = self._create_input(diaids=diaids, speaker_mode=self.speaker_mode,
                                          num_past_utterances=self.num_past_utterances,
                                          num_future_utterances=self.num_future_utterances)

    def __getitem__(self, index):

        return self.inputs_[index]


if __name__ == '__main__':
    ds_test = ErcTextDataset(DATASET='DailyDialog', SPLIT='test',
                             num_past_utterances=1000, num_future_utterances=1000,
                             model_checkpoint='roberta-base',
                             ROOT_DIR="../data", SEED=42)
    a = ds_test.inputs_
    #print(prompt_get_emotion2id('MELD'))
