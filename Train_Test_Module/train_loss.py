import torch


def mean_loss(output, 
              label,
              args,
    ):

    for i in range(args.batch_size):
        label_int = int(label[i])
        output[i, label_int] -= 1

    return torch.mean(output**2)