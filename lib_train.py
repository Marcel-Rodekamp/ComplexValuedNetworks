import torch
from tqdm.auto import tqdm 

def train(epochs, NN, loss, optimizer, trainData, validData):
    lossTrain = torch.zeros( epochs )
    lossValid = torch.zeros( epochs )

    for epoch in tqdm(range(epochs)):
        # get a mini batch from the training data
        for phiMiniBatch, in trainData:
            # Forward pass: prediction
            phiM, logDetJ_NN = NN(phiMiniBatch)
        
            # Forward pass: loss
            lossValue = loss(phiM, logDetJ_NN)
            
            # zero out the last gradient step
            optimizer.zero_grad()

            # backproagate to get the gradients
            lossValue.backward()

            # update the weights of NN
            optimizer.step() 

        # store the loss for post processing
        lossTrain[epoch] = lossValue.item()

        # evaluate the current state on the validation data
        with torch.no_grad():
            phiM, logDetJ_NN = NN(validData)

            lossValue = loss(phiM,logDetJ_NN)

            lossValid[epoch] = lossValue.item()
            
    return NN, lossTrain, lossValid


