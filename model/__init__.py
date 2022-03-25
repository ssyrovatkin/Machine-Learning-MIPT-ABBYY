import SciDevNN.SDmodules as SDModules
from SciDevNN.SDbase import Model
import torch.utils.data
import matplotlib.pyplot as plt

class LogisticRegression(Model):
    def __init__(self, loss=None, optimizer=None):
        super(LogisticRegression, self).__init__()

        self.loss = loss
        self.optimizer = optimizer

        self.linear = SDModules.Linear(784, 10, optimizer=optimizer)

    def forward(self, x):

        x = self.linear(x)

        return x

    def step(self, x, y):
        self.zero_grad()
        loss = self.loss(x,y)
        grad = self.loss.backward(x, y)
        grad = self.linear.backward(grad)
        self.linear.apply_grad()

        return loss

    def zero_grad(self):
        self.linear.zero_grad()

    def train(self, data, n_epochs):

        train_data, test_data = data

        train_losses = []
        train_counter = []
        test_accuracy = []
        test_losses = []

        test_counter = [num * train_data.num_images for num in range(n_epochs)]

        training_batch_size = 100
        test_batch_size = 1000

        training_loader = torch.utils.data.DataLoader(train_data, batch_size=training_batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=True)

        for epoch in range(1, n_epochs + 1):

            for batch_idx, (images, labels) in enumerate(training_loader):
                output = self.forward(images)
                current_loss = self.step(output, labels)

                if batch_idx % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * training_batch_size, len(train_data),
                               100 * batch_idx / len(training_loader), current_loss))
                if batch_idx % 600 == 0:
                    train_losses.append(current_loss)
                    train_counter.append((batch_idx * training_batch_size) + ((epoch - 1) * len(train_data)))

            test_loss = 0
            correct_guesses = 0

            for images, labels in test_loader:
                output = self.forward(images)
                test_loss += self.loss(output, labels)
                guesses = torch.max(output, 1, keepdim=True)[1]
                correct_guesses += torch.eq(guesses, labels.data.view_as(guesses)).sum()
            test_loss = test_loss / len(test_loader)
            test_losses.append(test_loss.detach())

            current_accuracy = float(correct_guesses) / float(len(test_data))
            test_accuracy.append(current_accuracy)

            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct_guesses, len(test_data),
                100. * current_accuracy))

        print('Total epochs: {}'.format(n_epochs))
        print('Max Accuracy is: {}%'.format(round(100 * max(test_accuracy), 2)))

        print(train_counter)
        print(test_counter)

        fig = plt.figure()
        plt.plot(train_counter, train_losses, color='blue', zorder=1)

        plt.scatter(test_counter, test_losses, color='red', zorder=2)

        plt.xlabel('number of training examples seen')
        plt.ylabel('negative log likelihood loss')
        fig.show()
        plt.pause(1000)

class SimpleCNN(Model):
    def __init__(self, loss=None, optimizer=None):
        super(SimpleCNN, self).__init__()

        self.loss = loss
        self.optimizer = optimizer

        self.conv1 = SDModules.Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1,
                                       optimizer=optimizer)
        self.linear = SDModules.Linear(784, 10, optimizer=optimizer)

    def forward(self, x):

        x = self.conv1(x)
        x = x.reshape(-1, 784)
        x = self.linear(x)

        return x

    def train_step(self, x, y):
        self.zero_grad()
        loss = self.loss(x,y)
        grad = self.loss.backward(x, y)
        grad = self.linear.backward(grad)
        self.linear.apply_grad()
        grad = grad.reshape(-1, 1, 28, 28)
        grad = self.conv1.backward(grad)
        self.conv1.apply_grad()

        return loss

    def zero_grad(self):
        self.conv1.zero_grad()
        self.linear.zero_grad()

    def train(self, data, n_epochs):

        train_data, test_data = data

        train_losses = []
        train_counter = []
        test_accuracy = []
        test_losses = []

        test_counter = [num * train_data.num_images for num in range(n_epochs)]

        training_batch_size = 10
        test_batch_size = 1000

        training_loader = torch.utils.data.DataLoader(train_data, batch_size=training_batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=True)

        for epoch in range(1, n_epochs + 1):

            for batch_idx, (images, labels) in enumerate(training_loader):
                output = self.forward(images)
                current_loss = self.train_step(output, labels)


                if batch_idx % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * training_batch_size, len(train_data),
                               100 * batch_idx / len(training_loader), current_loss))
                if batch_idx % 600 == 0:
                    train_losses.append(current_loss)
                    train_counter.append((batch_idx * training_batch_size) + ((epoch - 1) * len(train_data)))

            test_loss = 0
            correct_guesses = 0

            for images, labels in test_loader:
                output = self.forward(images)
                test_loss += self.loss(output, labels)
                guesses = torch.max(output, 1, keepdim=True)[1]
                correct_guesses += torch.eq(guesses, labels.data.view_as(guesses)).sum()
            test_loss = test_loss / len(test_loader)
            test_losses.append(test_loss.detach())

            current_accuracy = float(correct_guesses) / float(len(test_data))
            test_accuracy.append(current_accuracy)

            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct_guesses, len(test_data),
                100. * current_accuracy))

        print('Total epochs: {}'.format(n_epochs))
        print('Max Accuracy is: {}%'.format(round(100 * max(test_accuracy), 2)))

        print(train_counter)
        print(test_counter)

        fig = plt.figure()
        plt.plot(train_counter, train_losses, color='blue', zorder=1)

        plt.scatter(test_counter, test_losses, color='red', zorder=2)

        plt.xlabel('number of training examples seen')
        plt.ylabel('negative log likelihood loss')
        fig.show()
        plt.pause(1000)