import os
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
import torch
from config import _BATCH_SIZE, _max_tokenizer_len
from config import frac_train, frac_test, frac_val
from src.data_utils.data_splitter import train_val_test
import numpy as np
import re

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

    def prepare_inputs(self, data_path:str):
        """function to write original data in model format csv

                Input:
                    data_path: (str) path of the cleaned Roberta format csv (basically the output of prepare_model_csv)
                Output:
                    train, val , test loaders
            """
        df = pd.read_csv(data_path)
        df_train, df_val, df_test =  train_val_test(df, stratify_colname='level', frac_train=frac_train, frac_val=frac_val, frac_test=frac_test)
        if os.path.isdir('datasets'):
            df_train.to_csv( "datasets/train.csv", index=False)
            df_val.to_csv("datasets/val.csv", index=False)
            df_test.to_csv("datasets/test.csv", index=False)
        else:
            raise FileNotFoundError("Need to add datasets Folder!")

        train, val, test = self.data_loader(train_file_path="datasets/train.csv", val_file_path="datasets/val.csv", test_file_path="datasets/test.csv")
        return train, val, test

    def robert_tokenize(self, text, padding=False):
        if padding:
            ## encode_plus: text -> token_ids, decode: tokens_ids -> text(Roberta),  tokenize: text(Roberta)-> tokens
            return self.tokenizer.tokenize(self.tokenizer.decode(self.tokenizer.encode_plus(text, add_special_tokens = True, max_length = _max_tokenizer_len, pad_to_max_length = True)['input_ids'], clean_up_tokenization_spaces=False))
        return self.tokenizer.tokenize(self.tokenizer.decode(self.tokenizer.encode_plus(text, add_special_tokens = True, max_length = _max_tokenizer_len)['input_ids'], clean_up_tokenization_spaces=False))

    def prepare_model_csv(self,data_path: str, out_path:str):
        """function to write original data in model format csv

                Input:
                    data_path: (str) path of the unclean/original file
                Output:
                    pd.DataFrame: A pandas dataframe object with columns:
                    id: (str) seeker_id+"_"+response_id
                    seeker_post: (str)
                    response_post: (str)
                    level: (int) Output classification label in [0,1,2]
                    rationale_labels: (np.ndarray) binary array of len = num tokens
                    rationale_labels_trimmed: (int) total number of roberta tokens to predict (without padding) [to be used as denominator in metric score calculations]
                    response_post_masked: (str) marks the rationale text extracted from response [not be used anywhere just for observations]
            """
        df = pd.read_csv(data_path)
        df.fillna("", inplace=True)
        _cols = ["id","seeker_post","response_post","level","rationale_labels","rationale_labels_trimmed","response_post_masked"]
        processed_df = pd.DataFrame(columns=_cols)
        for _idx, row in df.iterrows():
            seeker_post = row['seeker_post'].strip()
            response_post = row['response_post'].strip()
            response_token_ids = self.tokenizer.decode(self.tokenizer.encode_plus(response_post, add_special_tokens = True, max_length = _max_tokenizer_len, pad_to_max_length = True)['input_ids'], clean_up_tokenization_spaces=False)
            response_tokens = self.robert_tokenize(response_post, padding=True)
            response_non_padded_tokens = self.robert_tokenize(response_post)
            rationales = row['rationales'].strip().split("|")
            response_characters_words = np.zeros(len(response_post), dtype=int)
            rationale_labels = np.zeros(len(response_tokens), dtype=int)
            if len(response_tokens) != _max_tokenizer_len:
                raise ValueError("length of response token can't be grater than max_token_length")
                #continue
            posi = 0
            for idx in range(len(response_tokens)):
                # https://stackoverflow.com/questions/62422590/do-i-need-to-pre-tokenize-the-text-first-before-using-huggingfaces-robertatoken#
                if response_tokens[idx].startswith('Ġ'):
                    response_tokens[idx] = response_tokens[idx][1:]
                response_characters_words[posi:posi+len(response_tokens[idx])+1]=idx
                posi+=len(response_tokens[idx])+1
            
            response_masked = response_post
            if len(rationales) == 0 or row['rationales'].strip() == '':
                rationale_labels[1:len(response_non_padded_tokens)] = 1
                response_masked = ''
            
            for r in rationales:
                if r!="":
                    try:
                        rationale_token_ids = self.tokenizer.decode(self.tokenizer.encode(r, add_special_tokens = False))
                        match_idxs = re.search(rationale_token_ids, response_token_ids)
                        curr_match = response_characters_words[match_idxs.start(0):match_idxs.start(0)+len(rationale_token_ids)]
                        curr_match = list(set(curr_match))
                        curr_match.sort()

                        response_masked = response_masked.replace(r, ' ')
                        response_masked = re.sub(r' +', ' ', response_masked)

                        rationale_labels[curr_match] = 1
                    except:
                        continue
            rationale_labels_str = ','.join(str(x) for x in rationale_labels)
            rationale_labels_str_trimmed = ','.join(str(x) for x in rationale_labels[1:len(response_non_padded_tokens)])
            _id = row['sp_id']+"_"+row['rp_id']
            mini_df = pd.DataFrame([(_id, seeker_post, response_post, np.int(row['level']), rationale_labels_str, len(rationale_labels_str_trimmed), response_masked)], columns=_cols)
            processed_df = processed_df.append(mini_df)
        
        processed_df.to_csv(out_path, index=False)
