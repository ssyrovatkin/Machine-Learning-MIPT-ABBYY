import torch
from model import LogisticRegression
from model import SimpleCNN
from data.my_data import MNISTDataset
import SciDevNN.SDmodules as SDModules
import SciDevNN.SDoptim as SDoptim

def save_model(model):
    raise NotImplementedError

def train_model(model, data):
    model.train(data, 25)


if __name__ == "__main__":

    train_data = MNISTDataset('C:/Users/mnist/train-images-idx3-ubyte.gz', 'C:/Users/mnist/train-labels-idx1-ubyte.gz')
    test_data = MNISTDataset('C:/Users/mnist/t10k-images-idx3-ubyte.gz', 'C:/Users/mnist/t10k-labels-idx1-ubyte.gz')

    data = (train_data, test_data)

    loss = SDModules.CrossEntropy()
    optimizer = SDoptim.Adam(lr = 0.02, betas=(0.9, 0.99))
    model = SimpleCNN(loss, optimizer)
    train_model(model, data)
