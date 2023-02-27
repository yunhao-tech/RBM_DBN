from principal_RBM_alpha import *

class DBN():
    def __init__(self, config) -> None:
        self.layers = []
        for i in range(len(config)-1):
            layer = init_RBM(config[i], config[i+1])
            self.layers.append(layer)
        self.n_layers = len(self.layers)

def init_DBN(config):
    """Initiate the parameters in DBN

    Parameters
    ----------
    config : list of integers
        number of neurons in each layer

    Returns
    -------
    instance of DBN class
    """
    assert len(config) > 2, "Must provide more than 2 integer arguments!"
    assert all(isinstance(x, int) for x in config), "All inputs must be integer!"
    return DBN(config)

def train_DBN(data, dbn, epochs, lr, batch_size):
    """Training process of DBN

    Parameters
    ----------
    data : numpy.ndarray: (n_samples, n_neurons_visible)
        training data
    dbn : instance of DBN class
        a DBN structure
    epochs : int
        number of epochs
    lr : float
        learning rate
    batch_size : int
        number of samples in one batch

    Returns
    -------
    a trained DBN instance
    """
    for i in range(dbn.n_layers):
        dbn.layers[i], _ = train_RBM(data, dbn.layers[i], epochs, lr, batch_size, verbose=0)
        data = entree_sortie_RBM(data, dbn.layers[i])
    return dbn

def generer_image_DBN(dbn_trained, n_gibbs, n_images):
    """Generate image from randomly sampled data

    Parameters
    ----------
    dbn_trained : a trained DBN
    n_gibbs : int
        number of iterations in Gibbs sampling in last layer of DBN
    n_images : int
        number of images to generate
    show_image : bool, optional
        whether to show images during generation, by default True
    """
    v = generer_image_RBM(dbn_trained.layers[-1], n_gibbs, n_images)
    for i in reversed(range(dbn_trained.n_layers-1)):
        v_proba = sortie_entree_RBM(v, dbn_trained.layers[i])
        v = np.random.binomial(1, v_proba, size=v_proba.shape)
    
    return v

if __name__ == "__main__":
    import argparse
    import logging.config
    from utils import compare_DBN

    logging.config.fileConfig("logging.conf")
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='Generate images using a trained RBM')
    parser.add_argument('--n_characters', type=int, default=1,
                    help='Number of characters for training')
    parser.add_argument('--n_neurons', type=int, default=200,
                    help='Number of neurons in each layer')
    parser.add_argument('--n_layers', type=int, nargs='+',
                    help='A list of number of layers to test separately')
    
    args = parser.parse_args()
    compare_DBN(args.n_characters, args.n_neurons, args.n_layers, logger)

    # Close the FileHandler
    for hdle in logger.handlers[:]:
        if isinstance(hdle,logging.FileHandler): 
            hdle.close()
            logger.removeHandler(hdle)