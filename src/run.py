import numpy as np
import torch
from src.data_utils.data_loader import DataReaderUtility
from src.models.epitome import EPITOME
import logging
logging.getLogger().setLevel(logging.INFO)
from transformers import  AdamW
from src.utils.trainer import trainer, no_grad_run

import random
from config import _SEED_VAL, _LR, _EPS
from tqdm import tqdm
random.seed(_SEED_VAL)
np.random.seed(_SEED_VAL)
torch.manual_seed(_SEED_VAL)
torch.cuda.manual_seed_all(_SEED_VAL)

if __name__=="__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        logging.info('No GPU! Sad Day!')
        device = torch.device("cpu")
    train, val, test = DataReaderUtility().prepare_inputs()
    logging.info("Train, Test and Validation sets created")
    logging.info("Loading EPITOME MODEL")
    model = EPITOME()
    model = model.to(device)
    logging.info("MODEL LOADED")
    

    ### Disabling the seeker parameters learning
    for p in model.seeker_encoder.parameters():
	    p.requires_grad = False

    ### optiizer
    optimizer = AdamW(model.parameters(),lr = _LR, eps = _EPS)

    ### Training and Validation
    logging.info("Training and Validation started!")
    model = trainer(data=train, data_val=val, optimizer=optimizer, model=model, device=device)
    logging.info("Training and Validation Completed!")

    ### Testing
    logging.info("Test set evaluation started!")
    no_grad_run(model=model, data=test, task = 'test', device=device)
    logging.info("Test set evaluation completed!")

    print("DONE!!!")