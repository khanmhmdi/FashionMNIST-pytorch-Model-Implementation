from matplotlib import pyplot as plt
from torch import nn, optim

from src.DataLoader.Data import Data
from src.Model.ANN import ANN
from trainer import  trainer

if __name__ == "__main__":
    dataloader = Data(train_path="/content/fashion-mnist_train.csv",
                      test_path="/content/fashion-mnist_test.csv")
    train, test = dataloader.load_data()
    train, valid, test = dataloader.prepare_data(512)

    ann1 = ANN(dropout=False)
    LEARNING_RATE = 1e-2
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ann1.parameters(), lr=LEARNING_RATE)
    train_model = trainer(model=ann1, epoch=100, learning_rate=LEARNING_RATE, optimizer=optimizer, loss=loss_function)
    train_model.train(train, valid)
