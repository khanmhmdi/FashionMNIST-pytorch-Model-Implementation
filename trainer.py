import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F


class trainer:
    def __init__(self, model, epoch, learning_rate, optimizer, loss):
        """Initializing parameters."""
        self.model = model
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.criterion = loss
        self.train_total_step = None
        self.val_total_step = None

        self.val_loss = []
        self.val_acc = []
        self.train_loss = []
        self.train_acc = []

    def train(self, trainset, validset):
        self.train_total_step = len(trainset)
        self.val_total_step = len(validset)

        for epoch in range(self.epoch):
            train_running_loss = 0
            train_correct = 0
            train_total = 0

            val_running_loss = 0
            val_correct = 0
            val_total = 0

            loss, train_correct, train_running_loss, train_total = self.run_train_epoch(train_correct,
                                                                                        train_running_loss, train_total,
                                                                                        trainset)

            loss, val_correct, val_running_loss, val_total = self.valid(loss, val_correct, val_running_loss, val_total,
                                                                        validset)

        self.save_model_result(train_correct, train_running_loss, train_total, val_correct, val_running_loss, val_total)

        batch_loss = 0
        total_t = 0
        correct_t = 0
        print("Iteration: " + str(epoch + 1), " LOSS : ", loss)

    def run_train_epoch(self, train_correct, train_running_loss, train_total, trainset):
        for (inputs, label) in trainset:
            inputs, label = Variable(inputs.float()), Variable(label)

            inputs = F.normalize(inputs)
            outputs = self.model(inputs)

            self.optimizer.zero_grad()
            loss = self.criterion(outputs, label)

            loss.retain_grad()
            loss.backward()
            self.optimizer.step()

            train_running_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            train_correct += torch.sum(pred == label).item()
            train_total += label.size(0)
        return loss, train_correct, train_running_loss, train_total

    def save_model_result(self, train_correct, train_running_loss, train_total, val_correct, val_running_loss,
                          val_total):
        self.train_acc.append(100 * train_correct / train_total)
        self.train_loss.append(train_running_loss / self.total_step)
        self.val_acc.append(100 * val_correct / val_total)
        self.val_loss.append(val_running_loss / self.total_step)
        print(f'\ntrain loss: {np.mean(self.train_loss):.4f}, train acc: {(100 * train_correct / train_total):.4f}')
        print(f'\nvalidation loss: {np.mean(self.val_loss):.4f}, validation acc: {(100 * val_correct / val_total):.4f}')

    def valid(self, loss, val_correct, val_running_loss, val_total, validset):
        for (images, labels) in validset:
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            val_running_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            val_correct += torch.sum(pred == labels).item()
            val_total += labels.size(0)

            self.val_loss.append(loss)
            self.val_acc.append()
        return loss, val_correct, val_running_loss, val_total
