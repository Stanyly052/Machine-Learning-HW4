from __future__ import print_function
import argparse
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

#import self-defined modules below:
import Train_Test_Module as train_test_module
import models
import data_module



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-Dataset of Machine Learning Course HW4')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default = 100, metavar='N',
                        help='number of epochs to train (default: 30)')
                        
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--optimizer', default="Adam",
                        help='The optimizer used in this model')
    parser.add_argument('--model', default="CNN_No-NonLinearActivationFunction_100Epoch_LR0-001")

    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 3407)')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda")

    if "MLP" in args.model:
        model = models.MLP_7layers().to(device)
    elif "CNN" in args.model:
        model = models.CNN_7layers().to(device)
    else:
        print("wrong model input information")

    if args.optimizer == "Adam":    
        optimizer = optim.Adam(model.parameters(), 
                               lr = args.lr,
                    )
    else: 
        print("wrong optimizer input information")

    train_dataLoader, test_dataLoader = data_module.cifar_HW4(args.batch_size)

    train_loss_all_epochs = torch.zeros(args.epochs)
    for epoch in range(1, args.epochs + 1):
        average_loss_this_epoch = train_test_module.train(args, 
                                                          model, 
                                                          device, 
                                                          train_dataLoader, 
                                                          optimizer, 
                                                          epoch,
                                )

        train_loss_all_epochs[epoch-1] = average_loss_this_epoch

        train_test_module.test(model, device, test_dataLoader, args, epoch)


    plt.plot(train_loss_all_epochs.detach().numpy())
    plt.xlabel("Epoch Number")
    plt.ylabel("Train Loss")
    plt.savefig(f"/home/yuzhi/ML_HW_4/Experiment_Results/train_loss_LR{args.lr}_{args.optimizer}_{args.model}.png")

    torch.save(train_loss_all_epochs, f"/home/yuzhi/ML_HW_4/Experiment_Results/train_loss_all_epochs_LR{args.lr}_{args.optimizer}_{args.model}.pt")


if __name__ == '__main__':
    main()


