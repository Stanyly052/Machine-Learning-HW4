import torch

MLP_1_train_loss = torch.load("/home/yuzhi/ML_HW_4/Experiment_Results/train_loss_all_epochs_LR0.001_Adam_CNN_No-NonLinearActivationFunction_100Epoch_LR0-001.pt")

for i in range(10):
    #print(str(i) + "&", end=" ")
    print(str(round(float(MLP_1_train_loss[i]), 4)) + "&", end=" ")