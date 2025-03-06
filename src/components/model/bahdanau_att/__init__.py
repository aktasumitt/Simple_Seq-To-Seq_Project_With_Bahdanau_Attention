import torch
import torch.nn as nn


# Attention
class Bahdanau_Attention(nn.Module):
    
    def __init__(self,hidden_size):
        super(Bahdanau_Attention,self).__init__()
    
        self.encoder_w=nn.Linear(hidden_size,hidden_size)
        self.decoder_w=nn.Linear(hidden_size,hidden_size)
        self.score_w=nn.Linear(hidden_size,1)
        
    def forward(self,encoder_out,decoder_hidden_state):
        
        encoder_score=self.encoder_w(encoder_out)
        decoder_score=self.decoder_w(decoder_hidden_state.permute(1,0,2))
        
        score=nn.functional.tanh((encoder_score+decoder_score))
        
        linear_score=self.score_w(score)
        
        attn_weights=nn.functional.softmax(linear_score,1)
        
        context=attn_weights*encoder_out
        
        context_vector=torch.sum(context,1)
        
        return context_vector