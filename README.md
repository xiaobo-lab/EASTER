# EASTER
EASTER is a sentiment analysis tool of the paper: 
**[Incorporating Pre-trained Transformer Models into TextCNN for Sentiment Analysis on Software Engineering Texts]** .
EASTER (sEntiment Analysis on SE texts based on TExtcnn and RoBERTa) is an integrated training framework  which incorporate pre-trained transformer models into the sentence-classification oriented deep learning framework named TextCNN to better capture the unique expression of sentiments in SE texts. Specifically, we introduce an optimized BERT model named RoBERTa as the word embedding layer of TextCNN, along with additional residual connections between RoBERTa and TextCNN for better cooperation.
An empirical evaluation based on four datasets from different software information sites shows that our training framework can achieve the overall better accuracy and generalizability than four baseline approaches(SentiStrength, SentiStrength-SE, SESSION, and Senti4SD).
Since the benchmark datasets we use are not generated and released by ourselves, we do not provide them here. If you want to use any of them, you should fulfill the licenses that they were released with and consider citing the original papers, and you can download original datasets at [Senti4SD](https://github.com/collab-uniba/Senti4SD)and [Lin et.al@ICSE2018](https://sentiment-se.github.io/replication.zip).


## Overview
1. ```EASTER.py``` the codes of EASTER. 

2. ```roberta``` contains roberta model.

3. ```data/EASTER.h5``` the model weights file of EASTER


## Dependencies
1.python=3.8.10
2.tensorflow=2.7.0
3.transformers=4.15.0


## Running
The "test" method of EASTER.py is the entrance of prediction. And if you want to use it, you need to modify the path of the data file in the file. The "train" method can retrain the model.
