import torch
import tqdm
from src.exception.exception import ExceptionNetwork, sys


def model_validation(valid_dataloader, loss_fn, Model, device):
    try:
        Model.eval()

        progress_bar = tqdm.tqdm(range(len(valid_dataloader)), "Validation Process")

        with torch.no_grad():

            loss_value_valid = 0
            correct_value_valid = 0
            total_value_valid = 0

            for batch_valid, (encoder_in_valid, decoder_out_valid) in enumerate(valid_dataloader):

                encoder_in_valid = encoder_in_valid.to(device)
                decoder_out_valid = decoder_out_valid.to(device)

                out_list_valid = Model(encoder_in_valid)
                loss_valid = loss_fn(out_list_valid, decoder_out_valid.reshape(-1))

                out_list_valid=out_list_valid.reshape(decoder_out_valid.size(0),decoder_out_valid.size(1),-1)
                _, pred = torch.max(out_list_valid, 2)

                loss_value_valid += loss_valid.item()
                correct_value_valid += (pred == decoder_out_valid).sum().item()
                total_value_valid += (decoder_out_valid.size(0)* (decoder_out_valid.size(1)))
                
                progress_bar.update(1)

        total_loss = loss_value_valid/(batch_valid+1)
        total_acc = (correct_value_valid/total_value_valid)*100

        progress_bar.set_postfix({"valid_acc": total_acc,
                                  "valid_loss": total_loss})

        return total_loss, total_acc

    except Exception as e:
        raise ExceptionNetwork(e, sys)
