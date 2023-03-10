# RBM_DBN 

Implement RBM (Restricted Boltzmann machine), DBN (Deep belief network) and DNN (Deep neural network) from scratch. 

Gained a lot for the whole process of peoject: construct the model; train, save and deploy the model; manage the logging information; write the config files.

# How to use
- folder `logs` saves the logging files; 
- `models` contains the pretrained DNN models; 
- the images generated by RBM and DBN are saved in the folder `outputs`;
- `environement.yml` provides the required Python package dependencies;
- `logging.conf` provides the logging configurations;
- there are some useful functions in `utils.py`;
- **`principal_RBM_alpha.py`, `principal_DBN_alpha.py`, `principal_DNN_MNIST.py`** are three main scripts. You can use them in this way (you can also find them in `commands.txt`):
```bash
python principal_RBM_alpha.py --n_characters 12 --n_neurons 200 300

python principal_DBN_alpha.py --n_characters 1 --n_neurons 300 --n_layers 2 3

python principal_DNN_MNIST.py --n_samples 1000 --epochs 2 --neurons_per_layer 200 --n_layers 2 --no-pretraining
```
You can find more details through the argument `-h`. For example: `python principal_DNN_MNIST.py -h`.

- `automatic_call.py` allows you to automatically call three above scripts multiple times (in case of doing tests).
