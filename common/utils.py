import numpy as np
import torch
from torchinfo import summary


def get_mean_and_std_bulk(dataset):
    """
        Calculate mean and std values of dataset without DataLoader(Transform).
    Args:
        dataset (torch.utils.data.Dataset): Pytorch dataset instance

    Returns:
        mean (torch.Tensor): mean of input dataset
        std(torch.Tensor): std of input dataset
    """
    np_data = dataset.data
    tensor_data = torch.from_numpy(np_data.astype(np.float32))  # ndarray -> Tensor
    tensor_data = tensor_data.div(255)  # normalize 0-1
    mean = tensor_data.mean(dim=(0, 1, 2))  # same as torch.mean(tensor_data, dim=(0, 1, 2))
    std = tensor_data.std(dim=(0, 1, 2))  # same as torch.std(tensor_data, dim=(0, 1, 2))
    return mean, std


def get_mean_and_std_sequence(dataset):
    """
        Calculate mean and std values of dataset with DataLoader(Transform).
    Args:
        dataset (torch.utils.data.Dataset): Pytorch dataset instance

    Returns:
        mean (torch.Tensor): mean of input dataset
        std(torch.Tensor): std of input dataset
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    channels = 3  # R,G,B
    mean_sum = torch.zeros(channels)
    sqrt_sum = torch.zeros(channels)
    batch_num = len(dataloader)

    for data, _ in dataloader:  # data shape = (batch, channel, height, width)
        mean_sum += torch.mean(data, dim=[0, 2, 3])
        sqrt_sum += torch.mean(data ** 2, dim=[0, 2, 3])

    mean = mean_sum / batch_num
    std = (sqrt_sum / batch_num - mean ** 2) ** 0.5
    return mean, std


def output_cuda_info():
    """
        Output cuda/cudnn info to console.
    Returns:
        None
    """
    print('# show CUDA and cuDNN information')
    print(f'CUDA__available: {torch.cuda.is_available()}')
    print(f'CUDA___is_built: {torch.backends.cuda.is_built()}')
    if torch.cuda.is_available():
        print(f'CUDA____version: {torch.version.cuda}')
    print(f'CUDA__dev_count: {torch.cuda.device_count()}')
    print(f'CUDNN_available: {torch.backends.cudnn.is_available()}')
    if torch.backends.cudnn.is_available():
        print(f'CUDNN___version: {torch.backends.cudnn.version()}')


def output_summary(model, trainloader):
    """
        Output model info to console.
    Args:
        model (nn.Module): Pytorch network model instance
        trainloader (DataLoader): Pytorch dataloader instance

    Returns:
        None
    """
    dataiter = iter(trainloader)
    images, _ = dataiter.next()
    summary(model, input_size=images.size(),  # images.size() => ([batch_size, channel, height, width])
            col_names=['input_size', 'output_size', 'num_params', 'kernel_size'])


def output_confusion_matrix(model, testloader, device='cpu'):
    """
        Ouput confusion matrix to console.
    Args:
        model (nn.Module): Pytorch network model instance
        testloader (DataLoader): Pytorch dataloader instance
        device (str): device name

    Returns:
        None
    """
    num_classes = len(testloader.dataset.classes)
    tensor = torch.zeros(num_classes, num_classes)

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                tensor[t.long(), p.long()] += 1
    print(tensor)
