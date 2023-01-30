import json
import pickle
from transformers import AutoTokenizer

def load_label_speaker_vocab(dataset_name):
    label_vocab = pickle.load(open("./data/"+dataset_name+"/label_vocab.pkl", "rb"))
    speaker_vocab = pickle.load(open("./data/" + dataset_name + "/speaker_vocab.pkl", "rb"))
    return label_vocab, speaker_vocab


def build_input(dia, tokenizer, has_future, speakers):
    dia_len = len(dia)
    max_token = 512-2
    new_dia = []
    #dia_len =
    #input = ' </s></s> ' + utterances[offset] + ' </s></s> '


def load_data(dataset_name, split, tokenizer_checkpoint, has_future):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
    assert split in ['train', 'dev', 'test']
    raw_data = json.load(open("./data/"+dataset_name+"/"+split+"_data.json", "r"))
    label_vocab, speaker_vocab = load_label_speaker_vocab(dataset_name)
    train_data = []
    dev_data = []
    test_data = []
    if dataset_name == 'IEMOCAP':
        # https: // www.ssa.gov/oact/babynames/decades/century.html
        speakers = {'Ses01': {'Female': 'Mary', 'Male': 'James'},
                   'Ses02': {'Female': 'Patricia', 'Male': 'John'},
                   'Ses03': {'Female': 'Jennifer', 'Male': 'Robert'},
                   'Ses04': {'Female': 'Linda', 'Male': 'Michael'},
                   'Ses05': {'Female': 'Elizabeth', 'Male': 'William'}}
    else:
        speakers = speaker_vocab
    for dialogue in raw_data:
        utt_input = build_input(dialogue, tokenizer, has_future, speakers)





load_data("MELD", "test", "roberta-base", True)
#load_label_speaker_vocab("IEMOCAP")