import torch.nn as nn
import torch
from src.utils import save_obj
from src.exception.exception import ExceptionNetwork, sys
from src.entity.config_entity import ModelConfig
from src.components.model.bahdanau_att import Bahdanau_Attention


# Model
class SEQ_TO_SEQ_MODEL(nn.Module):
    def __init__(self,embedding_size,max_len,num_layers,pad_idx,hidden_size,device):
        super(SEQ_TO_SEQ_MODEL,self).__init__()
        
        self.max_len=max_len
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.device=device
        
        self.encoder_embedding=nn.Embedding(num_embeddings=num_layers+1,embedding_dim=embedding_size,padding_idx=pad_idx)
        self.decoder_embedding=nn.Embedding(num_embeddings=num_layers+1,embedding_dim=embedding_size,padding_idx=pad_idx) # padding+ start object

        self.encoder_layer=self.Encoder(hidden_size=hidden_size)
        self.decoder_layer=self.Decoder(hidden_size=hidden_size)
        
        self.Attention=Bahdanau_Attention(hidden_size=hidden_size)
        
        self.decoder_out_layer=nn.Linear(in_features=hidden_size,out_features=num_layers+1)
    
    
    def Encoder(self,hidden_size):
        encoder_layer=nn.LSTM(input_size=hidden_size,hidden_size=hidden_size,batch_first=True,num_layers=1)
        return encoder_layer
    
    def Decoder(self,hidden_size):
        decoder_layer=nn.LSTM(input_size=hidden_size*2,hidden_size=hidden_size,batch_first=True,num_layers=1)
        return decoder_layer
    
    
    def forward(self,input_encoder):
        try:
            output_decoder_feature_list=[]
            
            # Create initial tensors for model
            input_decoder=torch.zeros((input_encoder.shape[0],1),dtype=torch.int).to(self.device) # 0 is start item
            context_vector=torch.zeros((input_encoder.shape[0],self.hidden_size),dtype=torch.int).to(self.device)
            
            encoder_embed=self.encoder_embedding(input_encoder) # Embedding Encoder
            encoder_out,(encoder_hs,encoder_cs)=self.encoder_layer(encoder_embed) # Encoder Layer (LSTM)
            
            for i in range(self.max_len):
                
                decoder_embed=self.decoder_embedding(input_decoder) # Embedding Decoder
                decoder_context=torch.cat([decoder_embed,context_vector.unsqueeze(1)],dim=-1) # Concattinating context and decoder embedding
                
                out_decoder,(decoder_hs,_)=self.decoder_layer(decoder_context,(encoder_hs,encoder_cs)) # Decoder layer (LSTM)
                
                context_vector=self.Attention(encoder_out,decoder_hs) # Apply attnention func
                
                output_decoder_feature=self.decoder_out_layer(out_decoder.view(-1,1*self.hidden_size)) # Drop 2 size (batch_Size,1,hidden_size) ---> (batch_Size,hidden_size) --->(batch_szie,num_layers+1)
                output_decoder_feature_list.append(output_decoder_feature)
                
                output_soft=nn.functional.softmax(output_decoder_feature,dim=1)
                _,pred=torch.max(output_soft,dim=-1)
                
                input_decoder=pred.unsqueeze(1)
            return torch.stack(output_decoder_feature_list).permute(1,0,2).reshape(-1,self.num_layers+1)
        except Exception as e:
            raise ExceptionNetwork(e,sys)


def create_and_save_model(config:ModelConfig):
    
    model=SEQ_TO_SEQ_MODEL(embedding_size=config.embedding_size,
                           max_len=config.max_len,
                           num_layers=config.label_size,
                           pad_idx=0,
                           hidden_size=config.hidden_size,
                           device=config.device)
    
    save_obj(model,config.model_save_path)