import torch
from lib_trainAnalysis import pathnameNN

def train(epochs, NN, loss, optimizer, trainData, validData, hyperparameters):
    
    r"""
        \param: epochs, int
                - total number of training epochs
        \param: NN, sequential
                - the neural network being trained
        \param: loss, class
                - the loss function used to train the network
        \param: optimizer
        \param: trainData = torch.utils.data.DataLoader(
            dataset = torch.utils.data.TensorDataset(
                torch.zeros((trainBatchSize,HM.Nt,HM.Nx), **torchTensorArgs).uniform_(-trainEpsilon,trainEpsilon)
            ), 
            batch_size = trainMiniBatchSize, 
            shuffle    = True
        )

        \param: validData, torch.rand((validBatchSize,HM.Nt,HM.Nx),**torchTensorArgs)
        """
    
    # Create memory for the value taken by the loss function after each training epoch, both for the training data and the validation data.
    lossTrain = torch.zeros( epochs )
    lossValid = torch.zeros( epochs )
    lossValidError = torch.zeros( epochs )
    lossTrainError = torch.zeros( epochs )
    # train
    for epoch in range(epochs):
        # get a mini batch from the training data
        for phiMiniBatch, in trainData:
            # Forward pass: prediction
            NNtrainData, logDetJ_NN = NN(phiMiniBatch)
        
            # Forward pass: loss
            lossValue, lossError = loss(NNtrainData, logDetJ_NN)
            
            # zero out the last gradient step
            optimizer.zero_grad()

           
            try:
                 # backproagate to get the gradients
                lossValue.backward()

                # update the weights of NN
                optimizer.step() 
            except:
                print(f"Singular matrix at epoch: {epoch}, hyperparameters: ",hyperparameters)
        # store the loss for post processing
        lossTrain[epoch] = lossValue.item()
        lossTrainError[epoch] = lossError.item()
        
        # evaluate the current state on the validation data
        with torch.no_grad():
            NNvalidData, logDetValid = NN(validData)

            lossValue,lossError = loss(NNvalidData,logDetValid)

            lossValid[epoch] = lossValue.item()
            lossValidError[epoch] = lossError.item()
            
        if ((epoch+1) % 200 == 0) and (epoch != 0):
            hyperparameters['epoch'] = epoch+1
            PATH = f"NN equivariance{pathnameNN(hyperparameters)}"
            torch.save(NN.state_dict(), PATH)
            
    return NN, lossTrain, lossValid, lossTrainError, lossValidError, NNvalidData, logDetValid

def train2(epochs, NN, loss, optimizer, trainData, validData, hyperparameters):
    
    r"""
        \param: epochs, int
                - total number of training epochs
        \param: NN, sequential
                - the neural network being trained
        \param: loss, class
                - the loss function used to train the network
        \param: optimizer
        \param: trainData = torch.utils.data.DataLoader(
            dataset = torch.utils.data.TensorDataset(
                torch.zeros((trainBatchSize,HM.Nt,HM.Nx), **torchTensorArgs).uniform_(-trainEpsilon,trainEpsilon)
            ), 
            batch_size = trainMiniBatchSize, 
            shuffle    = True
        )

        \param: validData, torch.rand((validBatchSize,HM.Nt,HM.Nx),**torchTensorArgs)
        """
    
    # Create memory for the value taken by the loss function after each training epoch, both for the training data and the validation data.
    lossTrain = torch.zeros( epochs )
    lossValid = torch.zeros( epochs )
    lossValidError = torch.zeros( epochs )
    lossTrainError = torch.zeros( epochs )
    # train
    for epoch in range(epochs):
        # get a mini batch from the training data
        for phiMiniBatch, in trainData:
            # Forward pass: prediction
            NNtrainData, logDetJ_NN = NN(phiMiniBatch)
        
            # Forward pass: loss
            lossValue, lossError = loss(NNtrainData, logDetJ_NN)
            
            # zero out the last gradient step
            optimizer.zero_grad()

            # backproagate to get the gradients
            lossValue.backward()

            # update the weights of NN
            optimizer.step() 

        # store the loss for post processing
        lossTrain[epoch] = lossValue.item()
        lossTrainError[epoch] = lossError.item()
        
        # evaluate the current state on the validation data
        with torch.no_grad():
            NNvalidData, logDetValid = NN(validData)

            lossValue,lossError = loss(NNvalidData,logDetValid)

            lossValid[epoch] = lossValue.item()
            lossValidError[epoch] = lossError.item()
            
        if ((epoch+1) % 100 == 0) and (epoch != 0):
            hyperparameters['epoch'] = epoch+1
            PATH = f"Results/NN{pathname(hyperparameters)}"
            torch.save(NN.state_dict(), PATH)
            
    return NN, lossTrain, lossValid, lossTrainError, lossValidError, NNvalidData, logDetValid


