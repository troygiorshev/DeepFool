"""MNIST 500-150-10 FC NN"""
import torch    # pytorch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(28*28, 500)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(500, 150)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(150, 10)
        
    def forward(self, x):
        x = self.l1(x)
        x = self.relu1(x)
        x = self.l2(x)
        x = self.relu2(x)
        x = self.l3(x)
        return F.log_softmax(x)


def main():
    """Main"""
    # Import MNIST and preprocess
    # Following a mix of pytorch tutorials online
    n_epochs = 3
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    random_seed = 1
    torch.manual_seed(random_seed)  # Deterministic

    # 0.1307, 0.3081 are the global mean and std deviation respectively

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('../data/', train=True, download=False,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('../data/', train=False, download=False,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
        batch_size=batch_size_test, shuffle=True)

    # Let's see what these do
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data.shape)

    # Make the NN
    network = Net()
    print(network)
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)

    # For nice graphs later
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

    # Training Function
    def train(epoch):
        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            print(data.size())
            data = data.view(-1, 28*28)
            print(data.size())
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
                torch.save(network.state_dict(), '../models/MNIST/FC1/model.pth')
                torch.save(optimizer.state_dict(), '../models/MNIST/FC1/optimizer.pth')

    def test():
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.view(-1, 28*28)
                output = network(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    
    test()
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()


if __name__ == "__main__":
    main()
