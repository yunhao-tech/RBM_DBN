from principal_DBN_alpha import *
from principal_RBM_alpha import *
from utils import *
import time
import random
import pickle
import logging.config

from scipy.special import softmax
import torchvision
from torchvision import transforms

class DNN():
    def __init__(self, config) -> None:
        self.config = config
        self.dbn = init_DBN(config[:-1])
        self.classification_layer = init_RBM(config[-2], config[-1])
    
def init_DNN(config):
    """Initiate the parameters in DNN

    Parameters
    ----------
    config : list of integers
        number of neurons in each layer

    Returns
    -------
    instance of DNN class
    """
    assert len(config) > 2, "Must provide more than 2 integer arguments!"
    assert all(isinstance(x, int) for x in config), "All inputs must be integer!"
    return DNN(config)

def pretain_DNN(data, dnn, epochs, lr, batch_size):
    """Greedy layer-wise pretraining for DBN in the DNN

    Parameters
    ----------
    data : numpy.ndarray: (n_samples, n_neurons_visible)
        training data
    dnn : instance of DNN class
        a DNN structure
    epochs : int
        number of epochs
    lr : float
        learning rate
    batch_size : int
        number of samples in one batch

    Returns
    -------
    a trained DNN instance
    """
    dnn.dbn = train_DBN(data, dnn.dbn, epochs, lr, batch_size)
    return dnn

def calcul_softmax(data, rbm):
    """Compute softmax for output of RBM

    Parameters
    ----------
    data : numpy.ndarray: (n_samples, n_neurons_visible)
        input data
    rbm : instance of RBM class

    Returns
    -------
    numpy.ndarray: (n_samples, n_neurons_hidden)
    """
    x = data @ rbm.w + rbm.b
    return softmax(x, axis=1)

def entree_sortie_reseau(data, dnn):
    """Compute output of each layer in DNN

    Parameters
    ----------
    data : numpy.ndarray: (n_samples, n_neurons_visible)
        input data
    dnn : instance of DNN class

    Returns
    -------
    list
        list of outputs in each layer
    """
    list_output = []
    for layer in dnn.dbn.layers:
        h_proba = entree_sortie_RBM(data, layer)
        data = np.random.binomial(1, h_proba, size=h_proba.shape)
        list_output.append(data)
    list_output.append(calcul_softmax(data, dnn.classification_layer))
    return list_output


def retropropagation(dataset, dataset_test, dnn, epochs, lr, batch_size, pre_training, 
                     pre_trained_model="3_layer_200", logger=logging.getLogger()):
    """Training process of DNN

    Parameters
    ----------
    dataset : Dataset instance
        Training data
    dataset_test : Dataset instance
        Test data
    dnn : instance of DNN class
    epochs : int
        number of epochs
    lr : float
        learning rate
    batch_size : int
        number of samples in one batch
    pre_training : bool
        whether to pre-train the DNN
    pre_trained_model : str, optional
        the file name of pre-trained DNN model. Used to load or save model

    Returns
    -------
    dnn : a trained DNN
    """
    logger.info(f"Loading data matrix from dataset with {len(dataset)} samples...")
    data_matrix, labels = get_data_matrix_from_dataset(dataset)
    if pre_training:
        folder = "models"
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, pre_trained_model + f"_samples_{len(dataset)}" + ".pickle")
        if os.path.isfile(filepath):
            logger.info("Loading pretrained DNN model from " + filepath)
            with open(filepath, 'rb') as handle:
                dnn = pickle.load(handle)
        else:
            logger.info(f"Start pretraining DNN: epochs=100, lr={lr}, batch size={batch_size}")
            start_time = time.time()
            dnn = pretain_DNN(data_matrix, dnn, 100, lr, batch_size)
            logger.info(f"Pretraining finished! Consumed time: {(time.time()- start_time):.2f}s")
            with open(filepath, 'wb') as handle:
                pickle.dump(dnn, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Start fine-tuning DNN...")
    start_time = time.time()
    dnn = train_DNN(dataset, dnn, epochs, lr, batch_size, pre_training)
    logger.info(f"Fine-tuning finished! Consumed time: {(time.time()- start_time):.2f}s")

    err_rate = test_DNN(data_matrix, labels, dnn)
    data_matrix_test, labels_test = get_data_matrix_from_dataset(dataset_test)
    err_rate_test = test_DNN(data_matrix_test, labels_test, dnn)
    logger.info(f"After fine-tuning: error rate on training set = {err_rate * 100:.1f}% || " 
          f"error rate on test set = {err_rate_test * 100:.1f}%")

    return dnn


def test_DNN(data_matrix, labels, dnn):
    """Evaluate the DNN 

    Parameters
    ----------
    data_matrix: numpy.ndarray
        Feature matrix of input data
    labels: numpy.ndarray
        Corresponding targets 
    dnn : a trained DNN

    Returns
    -------
    Float
        Error rate of classification
    """
    outputs = entree_sortie_reseau(data_matrix, dnn)
    pred = np.argmax(outputs[-1], axis=1)
    err_rate = np.mean(pred != labels)
    return err_rate


if __name__ == "__main__":
    import warnings
    import argparse

    warnings.filterwarnings('ignore')
    logging.config.fileConfig("logging.conf")
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='(Pre-)Training process of DNN')
    parser.add_argument('--n_samples', type=int, default=None,
                    help='Number of samples randomly choosen for training')
    parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-1,
                    help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size for (pre-)training')
    parser.add_argument('--n_layers', type=int, default=2,
                    help='Number of hidden layers')
    parser.add_argument('--neurons_per_layer', type=int, default=200,
                    help='Number of neurons in each hidden layer')
    # parser.add_argument('--pretraining', type=bool, default=True,
    #                 help='Whether to pre-train the DNN (in a unsupervied way)')
    parser.add_argument('--pretraining', action='store_true', 
                        help='Pre-train the DNN')
    parser.add_argument('--no-pretraining', dest='pretraining', action='store_false', 
                        help='Not to pre-train the DNN')
    parser.set_defaults(pretraining=True)

    args = parser.parse_args()

    data_dir = 'data'
    # MNIST dataset
    dataset = torchvision.datasets.MNIST(root=data_dir,
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=True)

    dataset_test = torchvision.datasets.MNIST(data_dir, 
                                            train=False, 
                                            download=True, 
                                            transform=transforms.ToTensor())
    dataset_binarized = BinarizedMNIST(dataset)
    dataset_binarized_test = BinarizedMNIST(dataset_test)
    if args.n_samples is not None:
        indices = random.sample(range(len(dataset_binarized)), args.n_samples)
        logger.info(f"Randomly choose {args.n_samples} for training")
        dataset_binarized = torch.utils.data.Subset(dataset_binarized, indices)
    else:
        logger.info("Making use of all training data")

    logger.info(f"DNN config: Batch size={args.batch_size}; {args.n_layers} hidden layers, "
                  f"each with {args.neurons_per_layer} neurons.")
    config = [784] + [args.neurons_per_layer] * args.n_layers + [10]
    model_name = f"{args.n_layers}_layers_{args.neurons_per_layer}"
    dnn = init_DNN(config)
    dnn = retropropagation(dataset_binarized, dataset_binarized_test,
                        dnn, args.epochs, args.lr, args.batch_size,
                        pre_training=args.pretraining, pre_trained_model=model_name,
                        logger=logger)
    
    # Close the FileHandler
    for hdle in logger.handlers[:]:
        if isinstance(hdle,logging.FileHandler): 
            hdle.close()
            logger.removeHandler(hdle)