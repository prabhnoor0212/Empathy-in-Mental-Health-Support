import numpy as np
from src.data_utils.data_loader import DataReaderUtility
import unittest
import pandas as pd
import torch
from src.models.epitome import EPITOME
from transformers import  AdamW
from config import _EPS, _LR, _LAMBDA_EI, _LAMBDA_RE, _BATCH_SIZE, _max_tokenizer_len
import logging
logging.getLogger().setLevel(logging.INFO)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    logging.info('No GPU! Sad Day!')
    device = torch.device("cpu")

class Test_Suite(unittest.TestCase):
    """Class for unit test
    """
    @classmethod
    def setUpClass(cls):
        file_paths = ["datasets/emotional-reactions-reddit.csv", "datasets/explorations-reddit.csv", "datasets/interpretations-reddit.csv"]
        cls.data = []
        cls.out_paths = []

        ### data loaders
        for file_path in file_paths:
            file_name = file_path.split("/")[-1].split(".")[0]
            out_path = "TEST/data/"+file_name+"_model.csv"
            cls.out_paths.append(out_path)
            DataReaderUtility().prepare_model_csv(file_path,out_path)
            train, val, test = DataReaderUtility().prepare_inputs(data_path=out_path)
            cls.data.append([train,val, test])

        ### model
        cls.model = EPITOME()
        cls.model = cls.model.to(device)
        for p in cls.model.seeker_encoder.parameters():
            p.requires_grad = False


        cls.optimizer = AdamW(cls.model.parameters(),lr = _LR, eps = _EPS)
    
    def test_data_loading(self):
        """Test for data splits check
        """
        self.assertEqual(len(Test_Suite.data), 3)
        for empathy_data in Test_Suite.data:
            self.assertEqual(len(empathy_data), 3)
    
    def test_dimemsions(self):
        """Test for checking the dimensions of the pre-processed files.
        """
        original_data = []
        for file_path in Test_Suite.out_paths:
            original_data.append(pd.read_csv(file_path))

        for idx, empathy_data in enumerate(Test_Suite.data):
            N = 0
            for idx, split in enumerate(empathy_data):
                n_batches = len(split)
                n_rows_in_split = len(split.dataset)
                N += n_rows_in_split

                self.assertEqual(n_batches, np.ceil(n_rows_in_split/_BATCH_SIZE))

                n_cols = len(split.dataset[0])
                self.assertEqual(n_cols, 7)
            self.assertEqual(N, original_data[idx].shape[0])

    def test_dtype_sanity(self):
        """Test for data types of the processed files.
        """
        for empathy_data in Test_Suite.data:
            for split in empathy_data:
                for row in split.dataset:
                    self.assertEqual(row[0].shape[0], _max_tokenizer_len)
                    self.assertEqual(row[0].dtype, torch.int64)
                    self.assertEqual(row[1].shape[0], _max_tokenizer_len)
                    self.assertEqual(row[1].dtype, torch.int64)
                    self.assertEqual(row[2].shape[0], _max_tokenizer_len)
                    self.assertEqual(row[2].dtype, torch.int64)
                    self.assertEqual(row[3].shape[0], _max_tokenizer_len)
                    self.assertEqual(row[3].dtype, torch.int64)
                    self.assertEqual(row[4].numel(), 1)
                    self.assertEqual(row[4].dtype, torch.int64)
                    self.assertEqual(row[5].shape[0], _max_tokenizer_len)
                    self.assertEqual(row[5].dtype, torch.int64)
                    self.assertEqual(row[6].numel(), 1)
                    self.assertEqual(row[6].dtype, torch.int64)


    def test_training(self):
        """Test for checking the training. (Basically, checks if the model weights are getting updated after first iteration)
        """
        Test_Suite.model.train()
        Test_Suite.model.zero_grad()
        row = Test_Suite.data[0][0].dataset[0:1]
        loss, empathy_loss, rationale_loss, logits_empathy, logits_rationale = Test_Suite.model(seeker_input = row[0].to(device),
                                                            responder_input = row[2].to(device), 
                                                            seeker_attn_mask=row[1].to(device),
                                                            responder_attn_mask=row[3].to(device), 
                                                            class_label=row[4].to(device),
                                                            rationale=row[5].to(device),
                                                            len_rationale=None,
                                                            lambda_EI=_LAMBDA_EI,
                                                            lambda_RE=_LAMBDA_RE)

        loss.backward()
        Test_Suite.optimizer.step()

        Test_Suite.model.zero_grad()
        n_loss, n_empathy_loss, n_rationale_loss, n_logits_empathy, n_logits_rationale = Test_Suite.model(seeker_input = row[0].to(device),
                                                            responder_input = row[2].to(device), 
                                                            seeker_attn_mask=row[1].to(device),
                                                            responder_attn_mask=row[3].to(device), 
                                                            class_label=row[4].to(device),
                                                            rationale=row[5].to(device),
                                                            len_rationale=None,
                                                            lambda_EI=_LAMBDA_EI,
                                                            lambda_RE=_LAMBDA_RE)
        
        self.assertEqual(n_loss.item()!=0, True)
        self.assertEqual(n_empathy_loss.item()!=0, True)
        self.assertEqual(n_rationale_loss.item()!=0, True)
        self.assertEqual((n_logits_empathy.cpu().detach().numpy() != logits_empathy.cpu().detach().numpy()).all(), True)
        self.assertEqual((n_logits_rationale.cpu().detach().numpy() != logits_rationale.cpu().detach().numpy()).all(), True)

if __name__ == "__main__":
    Test_Suite.setUpClass()
    logging.info("Data Loaded! Started Tests")
    Test_Suite().test_data_loading()
    Test_Suite().test_dimemsions()
    Test_Suite().test_dtype_sanity()
    Test_Suite().test_training()
    logging.info("All Tests Passed! :) ")