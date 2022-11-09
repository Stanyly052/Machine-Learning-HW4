import torch
from torch.autograd import Variable
from torch import nn



def train(args, model, device, train_loader, optimizer, epoch):

    model.train()

    total_loss_this_epoch = torch.zeros(1000)

    criterion = nn.CrossEntropyLoss()

    for batch_idx, sample in enumerate(train_loader):
        assert sample[0].shape[0] == 64
        if "MLP" in args.model:
            data = Variable(sample[0].view(args.batch_size, -1))
            label = Variable(sample[1])
        elif "CNN" in args.model:
            data = Variable(sample[0])
            label = Variable(sample[1])
        else:
            raise NameError('A wrong model is used')

        data, label = data.to(device), label.squeeze().to(device) #size of data: (64, 3, 32, 32)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)

        total_loss_this_epoch[batch_idx] = loss

        loss.backward()
        optimizer.step()

        #print the result during training
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [Batch Index: {}]\tLoss: {:.6f}'.format(epoch, batch_idx , loss.item()))
    
    average_loss_this_epoch = total_loss_this_epoch[:batch_idx+1].mean()

    return average_loss_this_epoch