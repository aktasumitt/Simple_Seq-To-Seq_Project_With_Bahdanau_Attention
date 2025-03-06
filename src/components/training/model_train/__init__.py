import torch
import tqdm
from src.exception.exception import ExceptionNetwork,sys


def model_training(train_dataloader, optimizer, loss_fn, Model,device):
    try: 
        Model.train()
        
        progress_bar=tqdm.tqdm(range(len(train_dataloader)),"Train Progress")
        
        loss_value_train=0
        correct_value_train=0
        total_value_train=0
                
        
        for batch_train,(encoder_in_train,decoder_out_train) in enumerate(train_dataloader):
           
            encoder_in_train=encoder_in_train.to(device)
            decoder_out_train=decoder_out_train.to(device)
            
            optimizer.zero_grad()
            
            out_list_train=Model(encoder_in_train)
            loss_train=loss_fn(out_list_train,decoder_out_train.reshape(-1))
            loss_train.backward()
            optimizer.step()
            
            out_list_train=out_list_train.reshape(decoder_out_train.size(0),decoder_out_train.size(1),-1)
            _,pred=torch.max(out_list_train,2)
            print(out_list_train.shape)
            
            
            loss_value_train+=loss_train.item()
            correct_value_train+=(pred==decoder_out_train).sum().item()
            total_value_train+=(decoder_out_train.size(0)*(decoder_out_train.size(1)))
        
            progress_bar.update(1)
            
        total_loss = loss_value_train/(batch_train+1)
        total_acc = (correct_value_train/total_value_train)*100
        
        progress_bar.set_postfix({"train_acc":total_acc,
                                  "train_loss":total_loss})
                
        return total_loss, total_acc
    
    except Exception as e:
            raise ExceptionNetwork(e,sys)
    

        