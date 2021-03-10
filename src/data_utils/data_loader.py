import os
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
import torch
from config import _BATCH_SIZE, _max_tokenizer_len

### Data load utils
class DataReaderUtility:
    """This class has utility to directly read the input csv and generate the necessary inputs required for modelling
    """
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
    def _load_data(self, file_path: str):
        """function to load data csv

            Input:
                file_path: (str) path of the file
            Output:
                pd.DataFrame: A pandas dataframe object
        """
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
            return df
        else:
            raise FileNotFoundError("No file found at path: %s"%(file_path))
    def _input_mask(self, post_list: list):
        _tokenizer = self.tokenizer.batch_encode_plus(post_list, add_special_tokens=True,max_length=_max_tokenizer_len, pad_to_max_length=True)
        return torch.tensor(_tokenizer['input_ids']), torch.tensor(_tokenizer['attention_mask'])

    def _prepare_input(self, file_path: str, train_flag: bool):
        """function to generate inputs in format for modelling

            Input:
                file_path: (str) path of the file
                train_flag: (bool) True if dataframe is train_df
            Output:
                Input Dict: (dict) Inputs for modelling
                [seeker_ip_ids, seeker_attention_mask,response_ip_ids, response_attention_mask,labels, rationales, trim_rationale]
        """
        df = self._load_data(file_path)
        seeker_ip_ids, seeker_attention_mask = self._input_mask(df['seeker_post'])
        response_ip_ids, response_attention_mask = self._input_mask(df['response_post'])
        labels = torch.tensor(df['level'].astype(int).values)
        rationales = list(df['rationale_labels'].apply(lambda s: torch.tensor([int(i) for i in s.split(',')], dtype=torch.long)).values)
        rationales = torch.stack(rationales, dim=0)

        # if train_flag:
        #     inputs = [seeker_ip_ids, seeker_attention_mask,response_ip_ids, response_attention_mask,labels, rationales]
        # else:
        trim_rationales = torch.tensor(df['rationale_labels_trimmed'].astype(int).values)
        inputs =  [seeker_ip_ids, seeker_attention_mask,response_ip_ids, response_attention_mask,labels, rationales, trim_rationales]
        return inputs
    
    def data_loader(self, train_file_path, val_file_path=None, test_file_path=None):
        if train_file_path:
            data_train = TensorDataset(*self._prepare_input(train_file_path, train_flag=True))
            data_loader_train = DataLoader(data_train, sampler = RandomSampler(data_train), batch_size = _BATCH_SIZE)
        else:
            raise Exception("train path is mandatory input")
        if val_file_path:
            data_val = TensorDataset(*self._prepare_input(val_file_path, train_flag=False))
            data_loader_val = DataLoader(data_val, sampler = SequentialSampler(data_val), batch_size = _BATCH_SIZE)
        else:
            data_loader_val = None
        if test_file_path:
            data_test = TensorDataset(*self._prepare_input(test_file_path, train_flag=False))
            data_loader_test = DataLoader(data_test, sampler = SequentialSampler(data_test), batch_size = _BATCH_SIZE)
        else:
            data_loader_test = None

        return data_loader_train, data_loader_val, data_loader_test