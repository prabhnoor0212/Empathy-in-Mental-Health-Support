### This file all the parameters and defaults

## config.py

#data configs
data_path = "datasets/emotional-reactions-reddit.csv" ## change this file to different datasets
train_path =  "dataset/train.csv"  #change this variable to run for custom train file
val_path="dataset/val.csv"         #if None: don't perform validation
test_path="dataset/test.csv"       #if None: don't use test set
frac_train=0.75
frac_val=0.05
frac_test=0.2

## basic model config
_max_tokenizer_len = 64            #default 64
_dropout = 0.1                     #default = 0.1
_EPS = 1e-8                       #default = 1e-8
_LR = 2e-5                        #default = 2e-5
_BATCH_SIZE=32                    #default = 32
_SEED_VAL=12                      #default = 12
_EPOCHS = 4                       #default = 4
_LAMBDA_EI=1                      #default = 1
_LAMBDA_RE=0.5                    #default = 0.5

## Multi-Head Attention
_num_head = 1                     #default = 1
_attn_dropout = 0.1               #default = 0.1

## Synthesizer (stretch goal) config
_attn_type='add'  
_attn_concat_type = 'simple'


_synthesizer_type = None # default = 'dense' | options = 'dense' or None


