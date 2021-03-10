### This file all the parameters and defaults

## config.py
train_path =  "dataset/train.csv"  #change this variable to run for custom train file
val_path="dataset/val.csv"         #if None: don't perform validation
test_path="dataset/test.csv"       #if None: don't use test set
_max_tokenizer_len = 64            #default 64
_dropout = 0.1                     #default = 0.1
_EPS = 1e-8                       #default = 1e-8
_LR = 2e-5                        #default = 2e-5
_BATCH_SIZE=32                    #default = 32
_SEED_VAL=12                      #default = 12
_EPOCHS = 4                       #default = 4
_LAMBDA_EI=1                      #default = 1
_LAMBDA_RE=0.5                    #default = 0.5
