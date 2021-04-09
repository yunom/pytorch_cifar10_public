import time
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


def get_mean_and_std1(dataset):
    np_data = dataset.data  # ndarray
    np_data = np_data / 255.  # 0~1の範囲に標準化(画像のみ適用可)
    mean = np.mean(np_data, axis=(0, 1, 2))
    std = np.std(np_data, axis=(0, 1, 2))
    return np.round(mean, 8), np.round(std, 8)


def get_mean_and_std2(dataset):
    tensor_data = torch.from_numpy(dataset.data.astype(np.float32))  # ndarray -> Tensor
    tensor_data = tensor_data.div(255)  # 0~1の範囲に標準化(画像のみ適用可)
    mean = tensor_data.mean(dim=(0, 1, 2))  # torch.mean(tensor_data, dim=(0, 1, 2))と同じ
    std = tensor_data.std(dim=(0, 1, 2))  # torch.std(tensor_data, dim=(0, 1, 2))と同じ
    return mean, std


def get_mean_and_std3(dataset):
    # webから引用(https://discuss.pytorch.org/t/mean-and-std-for-a-custom-dataset-olivetti/88793/6)
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


def get_mean_and_std4(dataset):
    # webから引用(https://github.com/kuangliu/pytorch-cifar)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def get_mean_and_std5(dataset):
    # webから引用(https://discuss.pytorch.org/t/mean-and-std-for-a-custom-dataset-olivetti/88793)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False, num_workers=2)
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, _ in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
    mean /= nb_samples
    std /= nb_samples
    return mean, std


def get_mean_and_std6(dataset):
    # https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/38
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    channels = 3
    mean = torch.zeros(channels)
    std = torch.zeros(channels)
    for i, data in enumerate(dataloader):
        data = data[0].squeeze(0)
        if i == 0:
            size = data.size(1) * data.size(2)
        mean += data.sum((1, 2)) / size
    mean /= len(dataloader)
    mean_dim = mean.unsqueeze(1).unsqueeze(2)

    for data in dataloader:
        data = data[0].squeeze(0)
        std += ((data - mean_dim) ** 2).sum((1, 2)) / size
    std /= len(dataloader)
    std = std.sqrt()
    return mean, std


def get_mean_and_std_after_standardize(dataset, mean, std):
    np_data = dataset.data  # ndarray
    np_data = np_data / 255.  # 0~1の範囲に標準化(画像のみ適用可)
    np_data = (np_data - mean) / std  # 引数のmeanとstdで正規化
    mean = np.mean(np_data, axis=(0, 1, 2))
    std = np.std(np_data, axis=(0, 1, 2))
    return mean, std


def with_print_stats(func, *args):
    print(f'# {func.__name__}')
    start = time.time()
    mean, std = func(*args)
    process_time = time.time() - start
    if type(mean) == torch.Tensor:
        mean = mean.to('cpu').detach().numpy().copy()
        std = std.to('cpu').detach().numpy().copy()
    print(f'mean:{mean}, std:{std}')
    print(f'processed time:{process_time: .2f}sec')
    return mean, std


if __name__ == '__main__':
    transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform)

    # オリジナルのデータセットの統計(全画像に対する一括行列計算)
    mean_1, std_1 = with_print_stats(get_mean_and_std1, trainset)  # numpyによる実装
    with_print_stats(get_mean_and_std2, trainset)  # Tensorによる実装

    # DataLoaderを経由してTransformされたデータセットの統計(画像単位の逐次計算)
    with_print_stats(get_mean_and_std3, trainset)  # 正しい実装
    mean_4, std_4 = with_print_stats(get_mean_and_std4, trainset)  # 誤った実装（stdの計算が不適切）
    with_print_stats(get_mean_and_std5, trainset)  # webから引用,誤った実装
    with_print_stats(get_mean_and_std6, trainset)  # webから引用,正しい実装

    # 各結果により正規化した後のmean, std
    print(f'## mean:{mean_1}及びstd:{std_1}による正規化後')
    with_print_stats(get_mean_and_std_after_standardize, trainset, mean_1, std_1)
    print(f'## mean:{mean_4}及びstd:{std_4}による正規化後')
    with_print_stats(get_mean_and_std_after_standardize, trainset, mean_4, std_4)
