from json import dumps
import sys

import torch

from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.evaluate import get_confusion_matrix

from data import addition, MNIST_test, MNIST_train, MNIST_Images, add_reduce_dataset
from deepproblog.examples.MNIST.network import MNIST_Net
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model


method = "exact"
N = 1

dataset_size = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
assert (dataset_size > 0 and dataset_size <= 100), "dataset size must be in percent (between 0-100)"
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 15
dataset_name = str(dataset_size)

add_reduce_dataset(dataset_size, dataset_name)

train_set = addition(N, dataset_name)
test_set = addition(N, "test")

dataset_import = MNIST_Images(dataset_name)

network = MNIST_Net()

net = Network(network, 'mnist_net', batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

model = Model("models/addition.pl", [net])
if method == "exact":
    model.set_engine(ExactEngine(model), cache=True)

model.add_tensor_source(dataset_name, dataset_import)
model.add_tensor_source("test", MNIST_test)

loader = DataLoader(train_set, batch_size, False)
train = train_model(model, loader, epochs, log_iter=100, profile=0)
model.save_state("snapshot/" + dataset_name + ".pth")
train.logger.comment(dumps(model.get_hyperparameters()))
train.logger.comment(
    "Accuracy {}".format(get_confusion_matrix(model, test_set, verbose=1).accuracy())
)
train.logger.write_to_file("log/" + dataset_name)
