"""
Created by Zhaoyan @ UCL
"""
from datetime import datetime
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import src.util.processors as P

def torch_dataset_download_helper():
    """Call this function if you want to download dataset via PyTorch API"""
    from six.moves import urllib

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]

    urllib.request.install_opener(opener)


def load_fmnist_classification(path_raw_dataset, train_size=-1, batch_size=1):
    """
    Fashion-MNIST

    Use half of the test set as validation set.

    :param train_size:
    :param path_raw_dataset:
    :param batch_size:
    :param measurement_size: (int) The length of vector of the compressed vector
    :param fix_seed: (bool) Whether the additive noise is deterministic in the dataset.
    :return: dict of pytorch Dataloader objects.
                {
                    'train':
                        (iterable) [measurement_vector, label_vector]
                        measurement_vector shape: [batch, measurement_size]
                        label_vector shape: [batch, 784]
                    'val':
                        (iterable) [measurement_vector, label_vector]
                        measurement_vector shape: [batch, measurement_size]
                        label_vector shape: [batch, 784]
                    'test':
                        (iterable) [measurement_vector, label_vector]
                        measurement_vector shape: [batch, measurement_size]
                        label_vector shape: [batch, 784]

                }
    """

    FashionMNIST = P.data_processor_wrapper(torchvision.datasets.FashionMNIST,
                                     P.Processor_Classification())
    transform_input = transforms.Compose([transforms.ToTensor()])
    try:
        data_train = FashionMNIST(root=path_raw_dataset, train=True, download=False, transform=transform_input)
    except:
        torch_dataset_download_helper()
        data_train = FashionMNIST(root=path_raw_dataset, train=True, download=True, transform=transform_input)

    try:
        data_val = FashionMNIST(root=path_raw_dataset, train=False, download=False, transform=transform_input)
    except:
        torch_dataset_download_helper()
        data_val = FashionMNIST(root=path_raw_dataset, train=False, download=True, transform=transform_input)

    if train_size != -1:
        data_train, _ = torch.utils.data.random_split(data_train,
                                                      [train_size, len(data_train)-train_size],
                                                      generator=torch.Generator().manual_seed(42))

    data_val, data_test = torch.utils.data.random_split(data_val,
                                                        [5000, 5000],
                                                        generator=torch.Generator().manual_seed(42))
    datasets = {'train': data_train, 'val': data_val, 'test': data_test}
    data_loaders = {i: torch.utils.data.DataLoader(datasets[i], batch_size=batch_size, shuffle=False)
                    for i in ['train', 'val', 'test']}
    return data_loaders


def load_mnist_classification(path_raw_dataset, train_size=-1, batch_size=1):
    """
    y = Ax + n
    n ~ N(mu, sigma^2)

    Use half of the test set as validation set.

    :param train_size:
    :param path_raw_dataset:
    :param batch_size:
    :param measurement_size: (int) The length of vector of the compressed vector
    :param fix_seed: (bool) Whether the additive noise is deterministic in the dataset.
    :return: dict of pytorch Dataloader objects.
                {
                    'train':
                        (iterable) [measurement_vector, label_vector]
                        measurement_vector shape: [batch, measurement_size]
                        label_vector shape: [batch, 784]
                    'val':
                        (iterable) [measurement_vector, label_vector]
                        measurement_vector shape: [batch, measurement_size]
                        label_vector shape: [batch, 784]
                    'test':
                        (iterable) [measurement_vector, label_vector]
                        measurement_vector shape: [batch, measurement_size]
                        label_vector shape: [batch, 784]

                }
    """

    MNIST = P.data_processor_wrapper(torchvision.datasets.MNIST,
                                     P.Processor_Classification())
    transform_input = transforms.Compose([transforms.ToTensor()])
    try:
        data_train = MNIST(root=path_raw_dataset, train=True, download=False, transform=transform_input)
    except:
        torch_dataset_download_helper()
        data_train = MNIST(root=path_raw_dataset, train=True, download=True, transform=transform_input)

    try:
        data_val = MNIST(root=path_raw_dataset, train=False, download=False, transform=transform_input)
    except:
        torch_dataset_download_helper()
        data_val = MNIST(root=path_raw_dataset, train=False, download=True, transform=transform_input)

    if train_size != -1:
        data_train, _ = torch.utils.data.random_split(data_train,
                                                      [train_size, len(data_train)-train_size],
                                                      generator=torch.Generator().manual_seed(42))

    data_val, data_test = torch.utils.data.random_split(data_val,
                                                        [5000, 5000],
                                                        generator=torch.Generator().manual_seed(42))
    datasets = {'train': data_train, 'val': data_val, 'test': data_test}
    data_loaders = {i: torch.utils.data.DataLoader(datasets[i], batch_size=batch_size, shuffle=False)
                    for i in ['train', 'val', 'test']}
    return data_loaders


def recipes(task_name, path_raw_dataset='', batch_size=1, dataset_kwargs=None, device=None):
    """

    :param task_name: (String) the name of the task. Select from:
                        'mnist_denoising'
                        'derain'
    :param path_raw_dataset: (optional, String)
    :param batch_size: (optional, int) batch size.
    :param kwargs: The kwargs for selected task.
    :return:
        'mnist_denoising'
            :key: 'train', 'val'
            iterable: input, (label, noise)
        'derain'
            :key: '

    """
    task_name = task_name.lower()

    if task_name == 'mnist_cl':
        """
        return:
        {
            'train': iterable: [inputs] [labels]
            'val:    iterable: [inputs] [labels]
            'test:   iterable: [inputs] [labels]
        }
        """
        dataloaders = load_mnist_classification(path_raw_dataset,
                                                train_size=dataset_kwargs['train_size'],
                                                batch_size=batch_size)

    elif task_name == f'fmnist_cl':
        """
        Fashion-MNIST classification
        return:
        {
            'train': iterable: [inputs] [labels]
            'val:    iterable: [inputs] [labels]
            'test:   iterable: [inputs] [labels]
        }
        """
        dataloaders = load_fmnist_classification(path_raw_dataset,
                                                 train_size=dataset_kwargs['train_size'],
                                                 batch_size=batch_size)
    else:
        exit(f'{datetime.now()} E load_dataset.recipes() Unknown task_name: {task_name}')
    return dataloaders
