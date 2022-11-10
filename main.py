from matplotlib import pyplot as plt
from torch import nn, optim

from src.DataLoader.Data import Data
from src.Model import *
from src.Model.ANN import *
from trainer import *

if __name__ == "__main__":
    dataloader = Data(train_path="/home/mohamadreza/PycharmProjects/ANN FIRST EXCERSIZE/Data/fashion-mnist_train.csv",
                      test_path="/home/mohamadreza/PycharmProjects/ANN FIRST EXCERSIZE/Data/fashion-mnist_test.csv")
    train, test = dataloader.load_data()
    train, valid, test = dataloader.prepare_data(512)

    ann = ANN()
    LEARNING_RATE = 1e-2
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ann.parameters(), lr=LEARNING_RATE)
    trainer = trainer(model=ann, epoch=100, learning_rate=LEARNING_RATE, optimizer=optimizer, loss=loss_function)
    trainer.train(train, valid)
