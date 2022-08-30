import isle.util
import itertools
import matplotlib.pyplot as plt
import torch
import copy

from pathlib import Path
from time import time_ns
from tqdm.auto import tqdm

from Misc import get_fn



def train(train_dataLoader, valid_dataLoader, model, optimizer, loss, param, pbar = None, torch_opt = {"use_gpu":False, "device":torch.device('cpu')}):
    train_loss = torch.zeros(size=(param['mini_batches'] , param['epochs']),device=torch_opt['device'],requires_grad=False)
    valid_loss = torch.zeros(size=(param['epochs'],),device=torch_opt['device'],requires_grad=False)

    own_pbar = pbar is None
    if own_pbar:
        pbar = tqdm(total = param['mini_batches']*param['epochs'])

    # train for the num_epochs
    e_train = 0
    e_valid = 0
    for epoch in range(param['epochs']):
        # ===========================
        # Train
        # ===========================
        s_train = time_ns() * 1e-9
        for i_mb,(mb_data, mb_data_flowed) in enumerate(train_dataLoader):
            if torch_opt['use_gpu']:
                # torch.cuda.synchronize()
                mb_data = mb_data.to(torch_opt['device'])
                mb_data_flowed = mb_data_flowed.to(torch_opt['device'])

            # zero the gradient data
            optimizer.zero_grad()
            # make a prediction
            pred,logDetJ = model(mb_data)
            # compute the loss
            loss_val = loss(pred, mb_data_flowed)
            # back propagate
            loss_val.backward()
            # update parameters
            optimizer.step()
            # average loss over all num_minibatches
            train_loss[i_mb,epoch] = loss_val.item()
            pbar.update(1)
        e_train += time_ns() * 1e-9 - s_train

        # ===========================
        # Valid
        # ===========================
        s_valid = time_ns() * 1e-9
        with torch.no_grad():
            for data,data_flowed in valid_dataLoader:
                if torch_opt['use_gpu']:
                    torch.cuda.synchronize()
                    data = data.to(torch_opt['device'])
                    data_flowed = data_flowed.to(torch_opt['device'])
                # make a prediction and compute loss to unseen data
                pred,logDetJ = model(data)
                valid_loss[epoch] = loss(pred, data_flowed).item()
        e_valid += time_ns() * 1e-9 - s_valid

        # output the train and validation time any 10th epoch
        if epoch % 10 == 9:
            print(f"Training epochs {epoch-9:02d}-{epoch:02d}|    "
                  + f"Train: "
                  + f"Time = {e_train:.2f}s, "
                  + f"Loss = {train_loss.mean(dim=0)[epoch].item():.4e}"
                  +  "    |    "
                  + f"Valid: Time = {e_valid:.2f}s, "
                  + f"Loss = {valid_loss[epoch].item():.4e}"
                  )
            # reset the timers
            e_train = 0
            e_valid = 0

    if own_pbar:
        pbar.close()

    return train_loss.mean(dim=0).detach(),valid_loss.detach()


def cross_validation(data, model, optimizer, loss, param):
    # checkup that the data is divisible into the required number of folds
    if(len(data) % param['folds'] != 0):
        raise ValueError(f"ERROR: number of folds ({param['folds']}) is not a divisor of training data set size ({len(data)}).")

    fold_data_size = len(data) // param['folds']

    # compute the mini batch size for the remaining data (removed one fold for validation)
    if (len(data)-fold_data_size) % param['mini_batches'] != 0:
        raise ValueError(f"ERROR: number of mini batches ({param['mini_batches']}) is not a devisor of reduced training data set size ({len(data)-fold_data_size}).")
    mb_data_size = (len(data) - fold_data_size) // param['mini_batches']

    # prepare output arrays: train_,valid_loss and train_,valid_variance
    train_loss = torch.zeros(param['folds'],param['epochs'])
    valid_loss = torch.zeros(param['folds'],param['epochs'])

    msg =f"lr={param['lr']}, "
    msg =f"lr = {param['lr']:.0e}, "
    msg+=f"epochs = {param['epochs']}, "
    msg+=f"mb = {param['mini_batches']}, "
    msg+=f"PRACL = {param['number_layers']}"

    # create a progress bar
    pbar = tqdm(total = param['mini_batches']*param['epochs']*param['folds'],desc=msg)

    # create a backup of the model to load it on every new CV step
    model_state_backup = copy.deepcopy(model.state_dict())

    for cv_step in range(param['folds']):
        model.load_state_dict(model_state_backup)

        # Identify validation data and prepare data loaders
        train_index_list = list(range(0,cv_step*fold_data_size))+list(range((cv_step+1)*fold_data_size,len(data)))
        valid_index_list = list(range(cv_step*fold_data_size,(cv_step+1)*fold_data_size))

        # pepare the data loader, no mini batches are taken at validation
        train_dataLoader = torch.utils.data.DataLoader(data,batch_size=mb_data_size,sampler=train_index_list)
        valid_dataLoader = torch.utils.data.DataLoader(data,batch_size=fold_data_size,sampler=valid_index_list)

        # perform training
        train_loss[cv_step,:],valid_loss[cv_step,:] = train(
            train_dataLoader=train_dataLoader,
            valid_dataLoader=valid_dataLoader,
            model=model,
            optimizer=optimizer,
            loss=loss,
            param=param,
            pbar = pbar
        )

        if param['save_model']:
            model_fn = get_fn(get_prependModel(param,cv_step,prepend="Model"),"pt",param,path=Path("Results/Models").absolute())
            optimizer_fn = get_fn(get_prependModel(param,cv_step,prepend="Optimizer"),"pt",param,path=Path("Results/Models").absolute())
            torch.save(model.state_dict(), model_fn)
            torch.save(optimizer.state_dict(), optimizer_fn)

    pbar.close()

    return train_loss,valid_loss


if __name__ == "__main__":
    pass
