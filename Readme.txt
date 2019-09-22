#Synthetic Collaborative Filtering (SynCF)

This archive contains a Python implementation of the SCFnet model 
under our proposed SynCF framework.
The framework aims at merging two significant pairs of perspectives
in Collaborative Filtering, namely representation learning vs. matching function
and low-level interactions vs. high-level interactions.


#FILES INCLUDED

SCFnet.py 
--- main file of SCFnet, one can directly run this file to train a SCFnet model. See parse_args() in the file for arguments.
PMF.py
--- implementation of Product Matrix Factorization, our proposed submodel for SCFnet aiming at high-level nonlinear interactions. 
     One can directly run this file to train a PMF model. See parse_args() in the file for arguments.
NCF.py
--- implementation of the method for Neural Collaborative Filtering in the paper
     He, X.; Liao, L.; Zhang, H.; Nie, L.; Hu, X.; and Chua, T.-S. 2017. Neural Collaborative Filtering. In WWW, 173¨C182.
     It is used as a submodel for SCFnet. One can directly run this file to train an NCF model. See parse_args() in the file for arguments.
Dataset.py
--- utils for processing datasets, implemented by Xiangnan He (xiangnanhe@gmail.com)
evaluate.py
--- utils for evaluating models, implemented by Xiangnan He (xiangnanhe@gmail.com)
Data
--- folder containing ml-1m and AMusic datasets, formerly provided in the Github project 
     DeepCF (https://github.com/familyld/DeepCF), in 2019, by
     Deng, Zhi-Hong and Huang, Ling and Wang, Chang-Dong and Lai, Jian-Huang and Yu, Philip S. 
     See below for details on their properties.
Pretrain
--- folder containing best ever pretrained SCFnet parameters on ml-1m and AMusic, accompanied with training descriptions.
     Notice pretrain_loader for the entire SCFnet has not been implemented in this version yet, but best ever parameters can 
     also be reached following configurations in the descriptions. 


#DATA DESCIPTIONS
                                                  num of users     num of items     num of ratings     sparsity
MovieLens 1 Million (ml-1m)          6400                  3706                 1000209          0.9553
Amazon Music (AMusic)                 1776                  12929                 46087            0.9980

train.rating - Each Line is a training instance: userID\t itemID\t rating\t timestamp (if have)
test.rating - Each Line is a testing instance: userID\t itemID\t rating\t timestamp (if have)
test.negative - Each line corresponds to the line of test.rating, containing 99 negative samples.
                       Each line is in the format: (userID,itemID)\t negativeItemID1\t negativeItemID2 ...


#EXAMPLES TO RUN THE CODES

To train a best ever SCFnet on ml-1m:
python SCFnet.py --dataset ml-1m --learner adam --PMFlayers2 [128]

To train a best ever SCFnet on AMusic:
python SCFnet.py --dataset AMusic --learner adam --commonitem [512,256] --commonuser [512,256]


#ENVIRONMENT SETTINGS

We use Keras with tensorflow as the backend.
The codes have been tested in both 
--- Python 3.6.4 with Keras 2.1.4 and tensorflow-gpu 1.7.0 
     on a laptop (Windows10 64bits, 8 Intel i7-6700HQ 2.60GHz CPUs, 8GB RAM, 8GB GTX 965m GPU)
--- Python 3.6.7 with Keras 2.2.4 and tesnsorflow 1.14.0rc1  
     in Google Colab (Ubuntu18.04 64bits, 2 Intel Xeon 2.20GHz CPUs, 13GB RAM,10GB Tesla K80 GPU)


#CONTACT

Cong Lin
Sun Yat-Sen University, P. R. China 
lincong_ceiling@126.com
linc7@mail2.sysu.edu.cn