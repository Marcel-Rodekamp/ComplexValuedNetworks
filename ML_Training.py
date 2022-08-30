import torch

from tqdm.auto import tqdm

from time import time_ns

def train(phaseStr, trainData, validData, model, optimizer, lossFct, params):
    trainLoss = torch.zeros(size = (params['Number Epochs'][phaseStr],))
    validLoss = torch.zeros(size = (params['Number Epochs'][phaseStr],))

    def trainStepSupervised(epoch):
        # get data from loader i.e. loop over minibatches
        for data,target in trainData:
            # zero the gradient data
            optimizer.zero_grad()
    
            # make a prediction
            pred,logDetJ = model(data)
            
            # compute the loss
            loss_val = lossFct(pred, target)

            # back propagate
            loss_val.backward()

            # update parameters
            optimizer.step()
            
            # average loss over all num_minibatches
            trainLoss[epoch] += loss_val.item()

        # normalize loss by minibatches
        trainLoss[epoch] /= len(trainData)

    def trainStepUnsupervised(epoch):
        # get data from loader i.e. loop over minibatches
        for data, in trainData:
            # zero the gradient data
            optimizer.zero_grad()
    
            # make a prediction
            pred,logDetJ = model(data)
            
            # compute the loss
            loss_val = lossFct(pred)

            # back propagate
            loss_val.backward()

            # update parameters
            optimizer.step()
            
            # average loss over all num_minibatches
            trainLoss[epoch] += loss_val.item()

        # normalize loss by minibatches
        trainLoss[epoch] /= len(trainData)

    def validStepSupervised(epoch):
        with torch.no_grad():
            # get data from loader i.e. loop over minibatches
            for data,target in validData:
                # make a prediction
                pred,logDetJ = model(data)
                
                # compute the loss
                loss_val = lossFct(pred, target)

                # average loss over all num_minibatches
                validLoss[epoch] += loss_val.item()

            # normalize loss by minibatches
            validLoss[epoch] /= len(validData)

    def validStepUnsupervised(epoch):
        with torch.no_grad():
            # get data from loader i.e. loop over minibatches
            for data, in validData:
                # make a prediction
                pred,logDetJ = model(data)
                
                # compute the loss
                loss_val = lossFct(pred)

                # average loss over all num_minibatches
                validLoss[epoch] += loss_val.item()

            # normalize loss by minibatches
            validLoss[epoch] /= len(validData)

    # define a progress bar
    pbar = tqdm(total = params['Number Epochs'][phaseStr])

    for epoch in range(params['Number Epochs'][phaseStr]):
        # ===============================================================
        # Train Step
        # ===============================================================
        trainTime = time_ns() * 1e-9

        if phaseStr == "RK4":
            trainStepSupervised(epoch)
        elif phaseStr == "FI":
            trainStepUnsupervised(epoch)
        else:
            raise RuntimeError("Couldn't determine training setup. Require 'phaseStr' to be either 'RK4' or 'FI' but got {phaseStr}!")

        trainTime = time_ns() * 1e-9 - trainTime

        # ===============================================================
        # Valid Step
        # ===============================================================
        validTime = time_ns() * 1e-9

        if phaseStr == "RK4":
            validStepSupervised(epoch)
        elif phaseStr == "FI":
            validStepUnsupervised(epoch)
        else:
            raise RuntimeError("Couldn't determine validation setup. Require 'phaseStr' to be either 'RK4' or 'FI' but got {phaseStr}!")

        validTime = time_ns() * 1e-9 - validTime

        # output the train and validation time any 10th epoch
        if epoch % 10 == 9:
            tqdm.write(f"Training epochs {epoch-9:04d}-{epoch:04d}|    "
                  + f"Train: "
                  + f"Time = {trainTime:.2f}s, "
                  + f"Loss = {trainLoss[epoch].item():.4e}"
                  +  "    |    "
                  + f"Valid: Time = {validTime:.2f}s, "
                  + f"Loss = {validLoss[epoch].item():.4e}"
            )
        elif (epoch == 0):
            tqdm.write(f"Training Starts with     |    "
                  + f"Train: "
                  + f"Time = {trainTime:.2f}s, "
                  + f"Loss = {trainLoss[epoch].item():.4e}"
                  +  "    |    "
                  + f"Valid: Time = {validTime:.2f}s, "
                  + f"Loss = {validLoss[epoch].item():.4e}"
            )

        # advance the progress bar
        pbar.update() 

    # close progress bar
    pbar.close()

    return trainLoss, validLoss 

if __name__ == "__main__":
    pass
