### This file all the parameters and defaults

## config.py



#data configs
data_path = "datasets/emotional-reactions-reddit.csv" ## change this file to different datasets [emotional-reactions-reddit.csv|explorations-reddit.csv|interpretations-reddit.csv]

# If you have pre-processed the data into the model formats only then touch the train/test/val parameters
train_path =  "dataset/train.csv"
val_path="dataset/val.csv"         
test_path="dataset/test.csv" 
frac_train=0.75
frac_val=0.05
frac_test=0.2

## Hyper-parameters
_max_tokenizer_len = 64           #default 64
_dropout = 0.1                    #default = 0.1
_EPS = 1e-8                       #default = 1e-8
_LR = 2e-5                        #default = 2e-5
_BATCH_SIZE=32                    #default = 32
_SEED_VAL=12                      #default = 12
_EPOCHS = 4                       #default = 4
_LAMBDA_EI=1                      #default = 1
_LAMBDA_RE=0.5                    #default = 0.5

## Multi-Head Attention (stretch goal)
_num_head = 1                     #default = 1
_attn_dropout = 0.1               #default = 0.1

## Synthesizer (stretch goal) config
_attn_type='add'                  # default = 'add' | options = 'add' or None
_attn_concat_type = 'simple'      # default = 'simple' | options = 'simple' or None
_synthesizer_type = None          # default = 'dense' | options = 'dense' or None
_synth_weight = 0.3               # default 0.3 | only active if synth_type is not None


_talking_heads = True          # default False | options = True or False
_n_taling_heads = 12           # default 12 | only active if _talking_heads is True
_talking_weight = 0.4          # default 0.3 | only active if _talking_heads is True