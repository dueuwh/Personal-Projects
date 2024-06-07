import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class Director:
    def __init__(self, model, optimizer, learning_rate, train_loader, test_loader, alpha_global=0.5, alpha_aux=0.5):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"{self.device} is selected for train/test")

        self.model = model
        self.model.to(self.device)
        self.optimizer = self.set_optimizer(optimizer, learning_rate)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.alpha_g = alpha_global
        self.alpha_a = alpha_aux
        self.train_loss = []
        self.test_loss = []

    def set_optimizer(self, optimizer, learning_rate):
        if optimizer == "SGD":
            return optim.SGD(params=self.model.parameters(), lr=learning_rate, momentum=0.9,
                             dampening=0, weight_decay=0, nesterov=False, maximize=False, foreach=None,
                             differentiable=False)
        elif optimizer == "Adam":
            return optim.Adam(params=self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08,
                              weight_decay=0, amsgrad=False, foreach=None, maximize=False, capturable=False,
                              differentiable=False)
        else:
            raise ValueError("Invalid optimizer")

    def loss_global(self, pred, label):
        output = 0
        return output

    def loss_aux(self, pred, label):
        output = 0
        return output

    def train(self):
        # “train” function should train self.model with self.train_loader.
        # index: 0, train_input.shape: torch.Size([4, 3, 32, 32]), label: tensor([5, 0, 5, 8])
        self.model.train()
        loss_step = 0
        for index, (train_input, label) in enumerate(self.train_loader):
            train_input, label = train_input.to(self.device), label.to(self.device)
            # print(f"index: {index}, train_input.shape: {train_input.shape}, label: {label}")
            output_global, output_aux = self.model(train_input)
            loss_g = self.loss_global(output_global, label)
            loss_a = self.loss_aux(output_aux, label)

            loss_total = self.alpha_g * loss_g + self.alpha_a * loss_a

            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()
            loss_step += loss_total.item()
        return loss_step/(index+1)

    def test(self):
        # “test” function should test self.model with self.test_loader.
        self.model.eval()
        loss_step = 0
        with torch.no_grad():
            for index, (test_input, label) in enumerate(self.test_loader):
                test_input, label = test_input.to(self.device), label.to(self.device)

                output_global, output_aux = self.model(test_input)

                loss_g = self.loss_global(output_global, label)
                loss_a = self.loss_aux(output_aux, label)

                loss_total = self.alpha_g * loss_g + self.alpha_a * loss_a

                loss_step += loss_total.item()
        return loss_step/(index+1)


    def plot(self, num_epoch):
        """
        The plot function uses matplotlib’s plt.plot() to create a line chart
        with epochs on the x-axis and accuracy(%) on the y-axis to display train and test result.
        The title must be your Id-number and name.
        """
        plt.title(f"202410024_남유상 | max epoch: {num_epoch}")
        plt.ylabel("Accuracy(%)")
        plt.xlabel("epochs")
        plt.plot(self.train_loss, color='blue', linestyle="solid", marker="p",  label="train loss")
        plt.plot(self.test_loss, color='red', linestyle="solid", marker="p", label="test loss")
        plt.legend()
        plt.show()

    def run(self, epochs):
        # If the “run” function is executed, it should repeat train and test for the number of epochs.
        for _ in range(epochs):
            self.train_loss.append(self.train()*100)
            self.test_loss.append(self.test()*100)

        self.plot(epochs)