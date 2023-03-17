import sys
from json import dumps

import torch

from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from network import Titanic_Net

from data import (
    Titanic_train, 
    Titanic_test,
    survive_pred,
)
from deepproblog.heuristics import geometric_mean
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.utils import get_configuration
from deepproblog.utils import format_time_precise
from deepproblog.utils import  config_to_string

epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 100
batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10


name = "titanic_DPL"
print(name)

train_set = survive_pred("train")
test_set = survive_pred("test")

network = Titanic_Net()

net = Network(network, "titanic_net", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

model = Model("models/titanic.pl", [net]) #currently almost empty 

model.set_engine(ExactEngine(model), cache=True)
    
model.add_tensor_source("train", Titanic_train)  
model.add_tensor_source("test", Titanic_test)

loader = DataLoader(train_set, batch_size, False)

train = train_model(model, loader, epochs, log_iter=500, profile=0)

model.save_state("snapshot/" + name + ".pth")

train.logger.comment(dumps(model.get_hyperparameters()))
train.logger.write_to_file("log/" + name)
