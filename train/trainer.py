import torch

class Trainer():
    def __init__(self, model, trainloader, num_epoch, optimizer, loss_function):
        self.model = model
        self.trainloader = trainloader
        self.num_epoch = num_epoch
        self.optimizer = optimizer
        self.loss_function = loss_function

    def train(self):
        for epoch in range(self.num_epoch):
            self.train_one_epoch(epoch)
        return self.model

    def train_one_epoch(self, epoch):
        running_loss = 0.0
        for i, data in enumerate(self.trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # TODO:
            # - change input to float32 when generate data
            # - change labels into one hot format when creating DataSet

            # forward + backward + optimize
            outputs = self.model(inputs.type(torch.float32))
            labels = self.binary_to_one_hot(labels) # convert labels into one hot format
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # print statistics
            running_loss += loss.item()
            once_in_a_while = 5
            if i % once_in_a_while == once_in_a_while - 1: # print out the loss every once in a while
                print('epoch %d, step %5d - loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / once_in_a_while))
                running_loss = 0.0

    def binary_to_one_hot(self, y):
        n = y.shape[0]
        one_hot = torch.zeros(n, 2)
        one_hot[:, 0][y == 1] = 1
        one_hot[:, 1][y == 0] = 1
        return one_hot

    def evaluate(self):
        pass