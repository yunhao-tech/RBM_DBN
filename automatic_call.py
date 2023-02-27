import subprocess

# automatically call `principal_RBM_alpha.py`
# for n_characters in [1, 3, 5, 7, 9]:
#     command = f"python principal_RBM_alpha.py --n_characters {n_characters} --n_neurons 100 400"
#     subprocess.call(command, shell=True)

# command = "python principal_RBM_alpha.py --n_characters 1 --n_neurons 100 200 400 800"
# subprocess.call(command, shell=True)


# # automatically call `principal_DBN_alpha.py`
# for n_characters in [1, 3, 5, 7, 9]:
#     command = f"python principal_DBN_alpha.py --n_characters {n_characters} --n_neurons 200 --n_layers 2"
#     subprocess.call(command, shell=True)

# for n_neurons in [100, 200, 400, 800]:
#     command = f"python principal_DBN_alpha.py --n_characters 1 --n_neurons {n_neurons} --n_layers 2"
#     subprocess.call(command, shell=True)

# command = "python principal_DBN_alpha.py --n_characters 1 --n_neurons 200 --n_layers 2 3 5 7"
# subprocess.call(command, shell=True)


# automatically call `principal_DNN_MNIST.py`
# for n_layers in [2, 3, 5]:
#     command = f"python principal_DNN_MNIST.py --epochs 200 --neurons_per_layer 200 --n_layers {n_layers} --pretraining"
#     subprocess.call(command, shell=True)
#     command = f"python principal_DNN_MNIST.py --epochs 200 --neurons_per_layer 200 --n_layers {n_layers} --no-pretraining"
#     subprocess.call(command, shell=True)

# for neurons_per_layer in [100, 300, 700]:
#     command = f"python principal_DNN_MNIST.py --epochs 200 --neurons_per_layer {neurons_per_layer} --n_layers 2 --pretraining"
#     subprocess.call(command, shell=True)
#     command = f"python principal_DNN_MNIST.py --epochs 200 --neurons_per_layer {neurons_per_layer} --n_layers 2 --no-pretraining"
#     subprocess.call(command, shell=True)

for n_samples in [1000, 3000, 7000, 10000, 30000]:
    command = f"python principal_DNN_MNIST.py --n_samples {n_samples} --epochs 200 --neurons_per_layer 200 --n_layers 2 --no-pretraining"
    subprocess.call(command, shell=True)
    command = f"python principal_DNN_MNIST.py --n_samples {n_samples} --epochs 200 --neurons_per_layer 200 --n_layers 2 --pretraining"
    subprocess.call(command, shell=True)

