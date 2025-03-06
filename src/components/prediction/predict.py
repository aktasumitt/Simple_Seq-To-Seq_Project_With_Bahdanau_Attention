from src.entity.config_entity import PredictionConfig
from src.utils import load_obj
from src.exception.exception import ExceptionNetwork,sys
import torch


class Prediction():
    
    def __init__(self,config:PredictionConfig):
        self.config=config
        self.model=load_obj(path=config.final_model_path).cpu()
        self.model.device="cpu"

    def prediction(self,input_data):
        try:
            
            # Padding_list
            input_pad_encoder=torch.cat([input_data,torch.zeros((self.config.max_len-len(input_data)),dtype=torch.int)]).unsqueeze(0)
                
            # Model Prediction
            with torch.no_grad():                
                out_list_test=self.model(input_pad_encoder)
                _,pred=torch.max(out_list_test,-1)
                    
            return pred
        except Exception as e:
           raise ExceptionNetwork(e,sys)
    
    