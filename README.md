**SCCL**

This repo contains our pytorch code for IEEE TAC accepted paper: "[Cluster-Level Contrastive Learning for Emotion Recognition in Conversations](http://arxiv.org/abs/2302.03508)". The architecture for our model is as follows:

![Image text](https://github.com/SteveKGYang/SCCL/blob/main/fig/SCCL.png)


**Preparation:**
1. Set up the Python 3.7 environment, and build the dependencies with the following code:
pip install -r requirements.txt

2. Download the data from https://drive.google.com/file/d/1b_ihQYKTAsO67I5LULMbMFBrDgat8bQN/view?usp=sharing.

3. Download the released pre-trained adapter model from the K-Adapter paper(https://arxiv.org/abs/2002.01808):
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


**Citation:**

Please cite our paper as follows:

@ARTICLE{10040720,
  author={Yang, Kailai and Zhang, Tianlin and Alhuzali, Hassan and Ananiadou, Sophia},
  journal={IEEE Transactions on Affective Computing}, 
  title={Cluster-Level Contrastive Learning for Emotion Recognition in Conversations}, 
  year={2023},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TAFFC.2023.3243463}}
