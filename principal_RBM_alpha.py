import numpy as np
import urllib.request
import scipy.io
import os.path
import matplotlib.pyplot as plt

def lire_alpha_digit(index=list(range(10, 11)), verbose=1):
    """Read the database Binary AlphaDigits

    Parameters
    ----------
    index : a list of index of characters,

    Returns
    -------
    numpy.ndarray
        Extracted data in matrix form: (n_samples, n_pixels)
    """
    filename = "binaryalphadigs.mat"
    folder = "data"
    if not os.path.exists(folder):
            os.makedirs(folder)
    filename = os.path.join(folder, filename)
    if not os.path.isfile(filename):
        url = "https://cs.nyu.edu/~roweis/data/binaryalphadigs.mat"
        urllib.request.urlretrieve(url, filename)
        if verbose > 0: print("Data Binary AlphaDigits is downloaded!")
    if verbose > 0: print("Loading data...")
    mat = scipy.io.loadmat(filename)['dat']
    if verbose > 0: print("Finished!")
    mat = mat[index]
    data = [x.flatten() for i in range(len(index)) for x in mat[i]]
    return np.array(data)

class RBM():
    def __init__(self, a, b, w) -> None:
        self.a = a
        self.b = b
        self.w = w

def init_RBM(p, q):
    """Initiate the parameters in RBM

    Parameters
    ----------
    p : int
        Number of neurons in visible layer
    q : int
        Number of neurons in hidden layer

    Returns
    -------
    instance of RBM class
    """
    a = np.zeros(p)
    b = np.zeros(q)
    w = np.random.normal(loc=0, scale=np.sqrt(0.01), size=(p, q))
    return RBM(a, b, w)

def entree_sortie_RBM(data_in, rbm):
    """From visible layer to hidden layer

    Parameters
    ----------
    data_in : numpy.ndarray: (n_samples, n_neurons_visible)
        visible data
    rbm : instance of RBM class
        a RBM structure

    Returns
    -------
    numpy.ndarray: (n_samples, n_neurons_hidden)
        p(h | data_in)
    """
    return 1 / (1 + np.exp(-(data_in @ rbm.w + rbm.b)))

def sortie_entree_RBM(data_hidden, rbm):
    """From hidden layer to visible layer

    Parameters
    ----------
    data_hidden : numpy.ndarray: (n_samples, n_neurons_hidden)
        hidden data
    rbm : instance of RBM class
        a RBM structure

    Returns
    -------
    numpy.ndarray: (n_samples, n_neurons_visible)
        p(v | data_hidden)
    """
    return 1 / (1 + np.exp(-(data_hidden @ rbm.w.T + rbm.a)))

def CD1(data, rbm):
    """Contrastive Divergence-1 algo

    Parameters
    ----------
    data : numpy.ndarray: (n_samples, n_neurons_visible)
        visible data
    rbm : instance of RBM class
        a RBM structure

    Returns
    -------
    h_proba_v0 : numpy.ndarray: (n_samples, n_neurons_hidden)
        p(h | data)
    v1 : numpy.ndarray: (n_samples, n_neurons_visible)
        reconstructed data after one round
    """
    h_proba_v0 = entree_sortie_RBM(data, rbm)
    h0 = np.random.binomial(1, h_proba_v0, size=h_proba_v0.shape)
    v_proba_h0 = sortie_entree_RBM(h0, rbm)
    v1 = np.random.binomial(1, v_proba_h0, size=v_proba_h0.shape)
    return h_proba_v0, v1

def check_reconstruction_loss(x, rbm):
    """Compute reconstruction loss per sample

    Parameters
    ----------
    x : numpy.ndarray: (n_samples, n_neurons_visible)
        visible data
    rbm : instance of RBM class
        a RBM structure

    Returns
    -------
    float
        quadratic loss between x and x_rec
    """
    _, x_rec = CD1(x, rbm)
    return np.sum((x - x_rec)**2) / x.shape[0]

def train_RBM(data, rbm, epochs, lr, batch_size, verbose=1):
    """Training process of RBM

    Parameters
    ----------
    data : numpy.ndarray: (n_samples, n_neurons_visible)
        training data
    rbm : instance of RBM class
        a RBM structure
    epochs : int
        number of epochs
    lr : float
        learning rate
    batch_size : int
        number of samples in one batch
    verbose : int, optional
        verbose > 0, print loss during training; else no info. By default 1

    Returns
    -------
    rbm : a trained RBM 

    losses: list
        loss of each epoch
    """
    losses = []
    n_batches = data.shape[0] // batch_size + 1
    for epoch in range(epochs):
        for i in range(n_batches - 1):
            v0 = data[i * batch_size : (i+1) * batch_size]
            h_proba_v0, v1 = CD1(v0, rbm)
            h_proba_v1 = entree_sortie_RBM(v1, rbm)
            grad_w = v0.T @ h_proba_v0 - v1.T @ h_proba_v1
            grad_a = np.sum(v0 - v1, axis=0)
            grad_b = np.sum(h_proba_v0 - h_proba_v1, axis=0)
            rbm.w = rbm.w + lr / v0.shape[0] * grad_w
            rbm.a = rbm.a + lr / v0.shape[0] * grad_a
            rbm.b = rbm.b + lr / v0.shape[0] * grad_b

        loss = check_reconstruction_loss(data, rbm)
        losses.append(loss)
        if verbose > 0:
            print(f"Epoch: {epoch:>3d} | Reconstruction loss = {loss:.2f}")
    return rbm, losses

def generer_image_RBM(rbm_trained, n_gibbs, n_images):
    """Generate image from randomly sampled data

    Parameters
    ----------
    rbm_trained : a trained RBM 
    n_gibbs : int
        number of iterations in Gibbs sampling
    n_images : int
        number of images to generate
    """
    imgs = []
    for _ in range(n_images):
        v = np.random.binomial(1, 0.5, size=rbm_trained.w.shape[0])
        for _ in range(n_gibbs):
            _, v = CD1(v, rbm_trained)
        imgs.append(v)
    
    return np.array(imgs)

if __name__ == "__main__":
    import argparse
    import logging.config
    from utils import compare_RBM

    logging.config.fileConfig("logging.conf")
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='Generate images using a trained RBM')
    parser.add_argument('--n_characters', type=int, default=1,
                    help='Number of characters for training')
    parser.add_argument('--n_neurons', type=int, nargs='+',
                    help='A list of number of hidden neurons to test separately')
    
    args = parser.parse_args()
    compare_RBM(args.n_characters, args.n_neurons, logger)

    # Close the FileHandler
    for hdle in logger.handlers[:]:
        if isinstance(hdle,logging.FileHandler): 
            hdle.close()
            logger.removeHandler(hdle)
    