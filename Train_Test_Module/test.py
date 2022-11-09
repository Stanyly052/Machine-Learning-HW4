import torch
from torch.autograd import Variable




def test(model, device, test_loader, args, epoch):

    model.eval()
    correct = 0.
    total = 0.
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):

            assert sample[0].shape[0] == 64
            if "MLP" in args.model:
                data = Variable(sample[0].view(args.batch_size, -1))
                label = Variable(sample[1])
            elif "CNN" in args.model:
                data = Variable(sample[0])
                label = Variable(sample[1])
            else:
                raise NameError('A wrong model is used')

            data, label = data.to(device), label.to(device)
            output = model(data)

            _, output_max_ind = torch.max(output, 1)
            correct_tensor = (output_max_ind == label).int()

            correct += correct_tensor.sum()
            total += data.shape[0]
            print('Test Accuracy of the model now: %f %%' % (100 * (correct.float() / total)))

        Final_Accuracy_This_Epoch = 100 * (correct.float() / total)

        with open("/home/yuzhi/ML_HW_4/Experiment_Results/Test_accuracy_LR{}_{}_{}.txt".format(args.lr, args.optimizer, args.model), 'a') as f:
            f.write(f"Epoch Number: {epoch}   " + str(Final_Accuracy_This_Epoch.detach().cpu().numpy()))
            f.write('\n')
