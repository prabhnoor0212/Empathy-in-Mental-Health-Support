import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
import math
import torch
import torch.nn.functional as F

from pre_trained_modeling.modeling_bert import BertEmbeddings, BertLayerNorm, BertModel, BertPreTrainedModel
from pre_trained_modeling.configuration_roberta import RobertaConfig
from pre_trained_modeling.roberta import RobertaForTokenClassification, RobertaModel


ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
	"roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
	"roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
	"roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
	"distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
	"roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
	"roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
}

input_index_map = {
    's_ip_ids':0,
    's_attn_mask':1,
    'r_ip_ids':2,
    'r_attn_mask':3,
    'class_label':4,
    'rationale':5,
    'len_rational':6
}

class Encoder(BertPreTrainedModel):
	config_class = RobertaConfig
	pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
	base_model_prefix = "roberta"

	def __init__(self, config):
		super().__init__(config)
		self.roberta = RobertaModel(config)
		self.init_weights() 


class EmpathyClassification(nn.Module):
    def __init__(self, embeddings_size=768, op_labels=3, empathy_dropout=0.1):
        super(EmpathyClassification, self).__init__()

        self.fcc_layer = nn.Linear(embeddings_size, embeddings_size)
        self.drop_layer = nn.Dropout(empathy_dropout)
        self.prediction_layer = nn.Linear(embeddings_size, op_labels)

    def forward(self, inputs):
        out = torch.relu(self.fcc_layer(self.drop_layer(inputs)))
        out = self.prediction_layer(self.drop_layer(out))
        return out

