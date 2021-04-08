import numpy as np
import torch
from src.data_utils.data_loader import DataReaderUtility
from src.models.epitome import EPITOME
import logging
logging.getLogger().setLevel(logging.INFO)
from transformers import  AdamW
from src.utils.trainer import trainer, no_grad_run
import argparse
parser = argparse.ArgumentParser("pathVars")
parser.add_argument("--model_write_path", default="output/out.pth", type=str, help="path where model will be written")


import random
from config import _SEED_VAL, _LR, _EPS, data_path
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

    
    args = parser.parse_args()
    out_path = data_path.split("/")
    out_path[-1] = out_path[-1].split(".csv")[0]+"_model.csv"
    out_path = "/".join(out_path)
    DataReaderUtility().prepare_model_csv(data_path, out_path)
    train, _, _ = DataReaderUtility().data_loader(train_file_path=out_path)
    logging.info("Loading EPITOME MODEL")
    model = EPITOME()
    model = model.to(device)
    logging.info("MODEL Initialized")
    

    ### Disabling the seeker parameters learning
    for p in model.seeker_encoder.parameters():
	    p.requires_grad = False

    ### optiizer
    optimizer = AdamW(model.parameters(),lr = _LR, eps = _EPS)

    ### Training 
    logging.info("Training started!")
    model = trainer(data=train, optimizer=optimizer, model=model, device=device)
    logging.info("Training Completed!")

    torch.save(model.state_dict(), args.model_write_path)

    logging.info("Model Saved!")


    logging.info("DONE!!!")