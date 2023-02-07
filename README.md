**SCCL**

This repo contains our pytorch code for IEEE TAC accepted paper: "Cluster-Level Contrastive Learning for Emotion Recognition in Conversations". The architecture for our model is as follows:

![Image text](https://github.com/SteveKGYang/SCCL/blob/main/fig/SCCL.png)


**Preparation:**
1. Set up the Python 3.7 environment, and build the dependencies with the following code:
pip install -r requirements.txt

2. Download the released pre-trained adapter model from the K-Adapter paper(https://arxiv.org/abs/2002.01808):
https://github.com/microsoft/k-adapter
and put the directory "fac-adapter" and "lin-adapter" under the directory ./SCCL/pretrained_models/.


**Training:**

You can train the model with the following codes:

Run on IEMOCAP with RoBERTa-Large:
python main.py --DATASET IEMOCAP --CUDA True --model_checkpoint roberta-large --alpha 0.8 --NUM_TRAIN_EPOCHS 5 --BATCH_SIZE 4

Run on IEMOCAP with FacAdapter:
python main.py --DATASET IEMOCAP --CUDA True --model_checkpoint roberta-facadapter --alpha 0.8 --NUM_TRAIN_EPOCHS 5 --BATCH_SIZE 4

Run on IEMOCAP with LinAdapter:
python main.py --DATASET IEMOCAP --CUDA True --model_checkpoint roberta-linadapter --alpha 0.8 --NUM_TRAIN_EPOCHS 5 --BATCH_SIZE 4

-------------------------------------------------------------------------------------------

Run on MELD with LinAdapter:
python main.py --DATASET MELD --CUDA True --model_checkpoint roberta-linadapter --alpha 1.0 --NUM_TRAIN_EPOCHS 3 --BATCH_SIZE 4

Run on MELD with FacAdapter:
python main.py --DATASET MELD --CUDA True --model_checkpoint roberta-facadapter --alpha 0.8 --NUM_TRAIN_EPOCHS 3 --BATCH_SIZE 4

Run on MELD with RoBERTa-Large:
python main.py --DATASET MELD --CUDA True --model_checkpoint roberta-large --alpha 1.0 --NUM_TRAIN_EPOCHS 3 --BATCH_SIZE 4

-------------------------------------------------------------------------------------------

Run on EmoryNLP with RoBERTa-Large:
python main.py --DATASET EmoryNLP --CUDA True --model_checkpoint roberta-large --alpha 0.8 --NUM_TRAIN_EPOCHS 5 --BATCH_SIZE 4

Run on EmoryNLP with FacAdapter:
python main.py --DATASET EmoryNLP --CUDA True --model_checkpoint roberta-facadapter --alpha 0.8 --NUM_TRAIN_EPOCHS 5 --BATCH_SIZE 4

Run on EmoryNLP with LinAdapter:
python main.py --DATASET EmoryNLP --CUDA True --model_checkpoint roberta-linadapter --alpha 0.8 --NUM_TRAIN_EPOCHS 5 --BATCH_SIZE 4

-------------------------------------------------------------------------------------------

Run on DailyDialog with RoBERTa-Large:
python main.py --DATASET DailyDialog --CUDA True --model_checkpoint roberta-large --alpha 0.8 --NUM_TRAIN_EPOCHS 5 --BATCH_SIZE 16

Run on DailyDialog with FacAdapter:
python main.py --DATASET DailyDialog --CUDA True --model_checkpoint roberta-facadapter --alpha 0.8 --NUM_TRAIN_EPOCHS 5 --BATCH_SIZE 16

Run on DailyDialog with LinAdapter:
python main.py --DATASET DailyDialog --CUDA True --model_checkpoint roberta-linadapter --alpha 0.8 --NUM_TRAIN_EPOCHS 5 --BATCH_SIZE 16