from src.exception.exception import ExceptionNetwork,sys
from src.logging.logger import logger
from src.entity.config_entity import DataIngestionConfig
import torch
from torch.utils.data import TensorDataset,random_split
from src.utils import save_obj

class DataIngestion():
    def __init__(self,config:DataIngestionConfig,TEST_MODE:bool=False):
        self.config=config
        self.TEST_MODE=TEST_MODE
        
    # extract zip file
    def create_lists(self,total_data_size,low_num=1,high_num=10,max_len=8):
    
        input_encoder_list=[]
        out_decoder_list=[]
        for _ in range(total_data_size):
            input_tensor=torch.randint(low=low_num,high=high_num+1,size=(torch.randint(1,max_len+1,(1,)).item(),),dtype=torch.int)
            input_encoder_list.append(input_tensor)
            
            reverse_list=input_tensor.flip(0)
            out_decoder_list.append(reverse_list)
                
        return input_encoder_list,out_decoder_list
    
    def padding(self,input_encoder,out_decoder,max_len=8):
        input_pad_encoder=[]
        out_pad_decoder=[]
        
        for tensor in input_encoder:
            input_pad=torch.cat([tensor,torch.zeros((max_len-len(tensor)),dtype=torch.int)])
            input_pad_encoder.append(input_pad)
        
        for tensor in out_decoder:
            out_decoder_pad=torch.cat([tensor,torch.zeros((max_len-len(tensor)),dtype=torch.int)])
            out_pad_decoder.append(out_decoder_pad)
    
        return torch.stack(input_pad_encoder),torch.stack(out_pad_decoder) 

    def transform_to_dataset(self,input,output):
        
        return TensorDataset(input,output.long())
    
    def random_split(self,dataset,test_rate,valid_rate):
        
        valid_size=int(len(dataset)*valid_rate)
        test_size=int(len(dataset)*test_rate)
        train_size=len(dataset)-(valid_size+test_size)
        
        train,valid,test=random_split(dataset,(train_size,valid_size,test_size))
        return train,valid,test
    
    def initialize_data_ingestion(self):
        try:
            
            input_encoder_list,out_decoder_list=self.create_lists(total_data_size=self.config.total_data_size)
            input_pad_encoder,out_pad_decoder=self.padding(input_encoder_list,out_decoder_list)
            dataset=self.transform_to_dataset(input_pad_encoder,out_pad_decoder)
            train_dataset,valid_dataset,test_dataset=self.random_split(dataset,test_rate=0.1,valid_rate=0.2)
            save_obj(train_dataset,self.config.transformed_train_dataset)   
            save_obj(valid_dataset,self.config.transformed_valid_dataset) 
            save_obj(test_dataset,self.config.transformed_test_dataset) 
               
        except Exception as e:
            raise ExceptionNetwork(e,sys)


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initialize_data_ingestion()
