import torch
import tqdm
from src.exception.exception import ExceptionNetwork,sys

    
def model_testing(test_dataloader, loss_fn, Model, devices="cpu"):
    try:    
        progress_bar = tqdm.tqdm(range(len(test_dataloader)), "Test Process")
        
        with torch.no_grad():
            
            progress_bar=tqdm.tqdm(range(len(test_dataloader)),"Test Progress")   
            loss_value_test=0
            correct_value_test=0
            total_value_test=0

                        
            for batch_test,(encoder_in_test,decoder_out_test) in enumerate(test_dataloader):
                            
                    encoder_in_test=encoder_in_test.to(devices)
                    decoder_out_test=decoder_out_test.to(devices)
                                                    
                    out_list_test=Model(encoder_in_test)
                    loss_test=loss_fn(out_list_test,decoder_out_test.reshape(-1))
                    
                    out_list_test=out_list_test.reshape(decoder_out_test.size(0),decoder_out_test.size(1),-1)  
                    _,pred=torch.max(out_list_test,2)
                            
                    loss_value_test+=loss_test.item()
                    correct_value_test+=(pred==decoder_out_test).sum().item()
                    total_value_test+=(decoder_out_test.size(0)*(decoder_out_test.size(1)))
                    
                    progress_bar.update(1)         
        
        progress_bar.set_postfix({"Test_Acc":100*(correct_value_test/total_value_test),
                                  "Test_Loss":(loss_value_test/batch_test+1)})

        total_loss = loss_value_test/(batch_test+1)
        total_acc = (correct_value_test/total_value_test)*100
        
        progress_bar.set_postfix({"valid_acc":total_acc,
                                  "valid_loss":total_loss})

        return total_loss, total_acc
    
    except Exception as e:
        raise ExceptionNetwork(e,sys)