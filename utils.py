import torch
import numpy as np
from torch.utils.data import Dataset
import os.path
import matplotlib.pyplot as plt
from principal_DBN_alpha import *
from principal_RBM_alpha import *

class BinarizedMNIST(Dataset):
    def __init__(self, dataset_mnist):
        super(BinarizedMNIST, self).__init__()
        self.data = dataset_mnist.data
        self.targets = dataset_mnist.targets

    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]
        img[img < 0.5] = 0
        img[img >= 0.5] = 1
        return img.float(), target

    def __len__(self):
        return len(self.targets)

def get_data_matrix_from_dataset(dataset):
    """Convert the Dataset (in Pytorch) to the numpy form

    Parameters
    ----------
    dataset : instance of Dataset class

    Returns
    -------
    data_matrix: numpy.ndarray
        Feature matrix of input data
    labels: numpy.ndarray
        Corresponding targets 
    """
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=256, 
                                            shuffle=False)
    data_matrix = []
    labels = []
    for data, label in data_loader:
        data_matrix.append(data)
        labels.append(label)
        
    data_matrix = torch.cat(data_matrix, dim=0)
    data_matrix = data_matrix.numpy().reshape(-1, 28*28)
    labels = torch.cat(labels, dim=0).numpy()
    return data_matrix, labels

def get_model(config):
    """Define a model in Pytorch 

    Parameters
    ----------
    config : list of integers
        Number of neurones in each layer

    Returns
    -------
    model in Pytorch
    """
    layers = []
    for i in range(len(config)-2):
        layer = torch.nn.Linear(config[i], config[i+1])
        activation = torch.nn.Sigmoid()
        layers.append(layer)
        layers.append(activation)
    layers.append(torch.nn.Linear(config[-2], config[-1]))
    model = torch.nn.Sequential(*layers)
    return model

def evaluate(model, data_loader):
    """Evaluate the model on dataset

    Parameters
    ----------
    model : a trained model in Pytorch
    data_loader : DataLoader instance in Pytorch

    Returns
    -------
    Float
        Cross Entropy loss
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    loss_total = 0
    for input, label in data_loader:
        input = input.reshape(-1, 28*28)
        output = model(input)
        loss_total += criterion(output, label) * input.shape[0]
    loss_total /= len(data_loader.dataset)
    return loss_total

def set_params_from_dnn_to_torchModel(dnn, model):
    """Set parameters of DNN to a model in Pytorch

    Parameters
    ----------
    dnn : instance of DNN class
    model : a model in Pytorch
    """
    with torch.no_grad():
        for i in range(dnn.dbn.n_layers):
            new_w = np.swapaxes(dnn.dbn.layers[i].w, 0, 1)
            model[2*i].weight = torch.nn.Parameter(torch.from_numpy(new_w).float())
            model[2*i].bias = torch.nn.Parameter(torch.from_numpy(dnn.dbn.layers[i].b).float())
        new_w = np.swapaxes(dnn.classification_layer.w, 0, 1)
        model[-1].weight = torch.nn.Parameter(torch.from_numpy(new_w).float())
        model[-1].bias = torch.nn.Parameter(torch.from_numpy(dnn.classification_layer.b).float())

def set_params_from_torchModel_to_dnn(dnn, model):
    """Set parameters of a model in Pytorch to DNN

    Parameters
    ----------
    dnn : instance of DNN class
    model : a model in Pytorch
    """
    with torch.no_grad():
        for i in range(dnn.dbn.n_layers):
            dnn.dbn.layers[i].w = np.swapaxes(model[2*i].weight.numpy(), 0, 1)
            dnn.dbn.layers[i].b = model[2*i].bias.numpy()
        dnn.classification_layer.w = np.swapaxes(model[-1].weight.numpy(), 0, 1)
        dnn.classification_layer.b = model[-1].bias.numpy()

def train_DNN(dataset, dnn, epochs, lr, batch_size, pre_training=True):
    """Train a Pytorch model, then convert it to a DNN instance

    Parameters
    ----------
    dataset : Dataset instance
        Training data
    dnn : instance of DNN class
    epochs : int
        number of epochs
    lr : float
        learning rate
    batch_size : int
        number of samples in one batch
    pre_training : bool, optional
        whether to load the parameters of dnn to Pytorch model, by default True

    Returns
    -------
    dnn : a trained DNN
    """
    model = get_model(dnn.config)
    model = model.float()
    if pre_training:
        set_params_from_dnn_to_torchModel(dnn, model)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=batch_size, 
                                            shuffle=True)

    for epoch in range(1, epochs+1):
        model.train()
        for input, label in data_loader:
            input = input.reshape(-1, 28*28)
            optimizer.zero_grad()
            output = model.forward(input)
            loss = criterion(output, label)
            loss.backward() 
            optimizer.step()
        loss_total = evaluate(model, data_loader)
        print(f"Epoch: {epoch:>3d} | Evaluation CE loss on training set : {loss_total:.2f}")

    set_params_from_torchModel_to_dnn(dnn, model)
    
    return dnn


def compare_RBM(n_characters, list_n_hidden_neurones, logger):

        all_data = lire_alpha_digit(index=list(range(10, 10+n_characters)), verbose=0)
        for n_hidden_neurones in list_n_hidden_neurones:
            rbm = init_RBM(p=all_data.shape[1], q=n_hidden_neurones)
            logger.info(f"Training RBM using {n_characters} characters: visible neurones={all_data.shape[1]}, "
                        f"hidden neurones={n_hidden_neurones}, epochs=200, lr=1e-1, batch size=16")
            rbm_trained, _ = train_RBM(all_data, rbm, 200, 1e-1, 16, verbose=0)

            n_images = 3
            imgs = generer_image_RBM(rbm_trained, 200, n_images)
            logger.info(f"Generated 3 images using the trained RBM, with 200 Gibbs iterations")

            fig, axes = plt.subplots(1, n_images, figsize=(12, 5))
            for i in range(n_images):
                axes[i].imshow(imgs[i].reshape(20, 16))
            fig.suptitle(f'RBM: hidden nodes {n_hidden_neurones}, {n_characters} character(s)', fontsize=15)
            plt.tight_layout()
            fig.subplots_adjust(top=0.9)
            folder = "outputs/RBM"
            if not os.path.exists(folder):
                os.makedirs(folder)
            plt.savefig(os.path.join(folder.split("/")[0], folder.split("/")[1],
                                    f'RBM_q_{n_hidden_neurones}_character(s)_{n_characters}.png'))
            logger.info(f"Images saved at the folder: " + folder)

def compare_DBN(n_characters, n_hidden_neurones, list_n_layers, logger):

    all_data = lire_alpha_digit(index=list(range(10, 10+n_characters)), verbose=0)
    for n_layers in list_n_layers:
        config = [320] + [n_hidden_neurones]*n_layers
        dbn = init_DBN(config)
        logger.info(f"Training DBN using {n_characters} characters: neurones in each layer={config}, "
                        f"epochs=200, lr=1e-1, batch size=16")
        dbn_trained = train_DBN(all_data, dbn, 200, 1e-1, 16)

        n_images = 3
        imgs = generer_image_DBN(dbn_trained, 200, n_images)
        logger.info(f"Generated 3 images using the trained DBN, with 200 Gibbs iterations")

        fig, axes = plt.subplots(1, n_images, figsize=(12, 5))
        for i in range(n_images):
            axes[i].imshow(imgs[i].reshape(20, 16))
        fig.suptitle(f'DBN: number of layers {n_layers}, hidden nodes {n_hidden_neurones}, {n_characters} character(s)', fontsize=15)
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)
        folder = "outputs/DBN"
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(os.path.join(folder.split("/")[0], folder.split("/")[1], 
                                 f'DBN_layers_{n_layers}_nodes_{n_hidden_neurones}_character(s)_{n_characters}.png'))
        logger.info(f"Images saved at the folder: " + folder)
