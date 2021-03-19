import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
import math
import torch
import torch.nn.functional as F

from src.models.attention import SelfAttention

from src.pre_trained_modeling.modeling_bert import BertLayerNorm, BertPreTrainedModel
from src.pre_trained_modeling.configuration_roberta import RobertaConfig
from src.pre_trained_modeling.roberta import RobertaModel

from config import _attn_concat_type, _attn_type, _synthesizer_type, _dropout, _attn_dropout, _num_head

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

class EPITOME(nn.Module):
    
    def __init__(self):
        super(EPITOME, self).__init__()

        

        ###attention
        self.self_attention = SelfAttention(heads = _num_head, attn_dropout=_attn_dropout)

        ###predictors
        self.empathy_classification = EmpathyClassification()
        self.rationale_classification = nn.Linear(768,2) ##emb size and binary class at token level

        ###droupout
        self.drop_layer = nn.Dropout(_dropout)

        self.apply(self._init_weights)

        ###encoders
        self.seeker_encoder = Encoder.from_pretrained("roberta-base",output_attentions = False,output_hidden_states = False)
        self.responder_encoder = Encoder.from_pretrained("roberta-base",output_attentions = False,output_hidden_states = False)
        

        if _synthesizer_type is not None and _synthesizer_type=='dense':
            ##stretch
            pass
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            initializer_range=0.02
            module.weight.data.normal_(mean=0.0, std=initializer_range)
        elif isinstance(module, BertLayerNorm):
        	module.bias.data.zero_()
        	module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, seeker_input, seeker_attn_mask, responder_input, responder_attn_mask, class_label, rationale, len_rationale,lambda_EI,lambda_RE):
        #seeker_input = seeker.to(long)
        seeker_token_embs = self.seeker_encoder.roberta(seeker_input,seeker_attn_mask)[0]
        response_token_embs = self.responder_encoder.roberta(responder_input,responder_attn_mask)[0]

        # seeker_token_embs = self.seeker(seeker_input,seeker_attn_mask)[0]
        # response_all_layers = self.responder(responder_input,responder_attn_mask)
        # response_token_embs = response_all_layers[0]
        context_token_embs = self.drop_layer(self.self_attention(response_token_embs,seeker_token_embs,seeker_token_embs))
        if _synthesizer_type is not None and _synthesizer_type=='dense':
            ### stretch goal
            pass
        else:
            if _attn_concat_type == 'simple':
                response_context_embs = response_token_embs+context_token_embs
            else:
                raise Exception("Invalid Context type")

        logits_empathy = self.empathy_classification(response_context_embs[:, 0, :])
        logits_rationales = self.rationale_classification(response_context_embs)
        
        outputs = (logits_empathy,logits_rationales)
        
        
        loss_rationales, loss_empathy = 0,0
        
        if rationale is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if responder_attn_mask is not None:
                active_loss = responder_attn_mask.view(-1) == 1
                active_logits = logits_rationales.view(-1, 2)
                active_labels = torch.where(active_loss, rationale.view(-1), torch.tensor(loss_fct.ignore_index).type_as(rationale))
                loss_rationales = loss_fct(active_logits, active_labels)
            else:
                loss_rationales = loss_fct(logits_rationales.view(-1, 2), rationale.view(-1))

        if class_label is not None:
            loss_fct = CrossEntropyLoss()
            loss_empathy = loss_fct(logits_empathy.view(-1, 3), class_label.view(-1))
            loss = lambda_EI * loss_empathy + lambda_RE * loss_rationales
        else:
            print("None label")

        outputs = (loss, loss_empathy, loss_rationales) + outputs
        return outputs
