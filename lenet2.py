import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import mnist
import matplotlib.pyplot as plt

class LeNetMod(nn.Module):

    def __init__(self, input_channels=1):
        super(LeNetMod, self).__init__()

        self.dropout=nn.Dropout(p=0.2)

        # C1: Conv Layer (input: 32x32 → output: 28x28)
        self.C1 = nn.Conv2d(in_channels=input_channels, out_channels=6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(6)
        
        # S2: Avg Pool (28x28 → 14x14)
        self.S2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # C3: Conv Layer (14x14 → 10x10)
        self.C3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        
        # S4: Avg Pool (10x10 → 5x5)
        self.S4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # C5: Fully connected convolution (5x5 input maps × 16 channels = 400)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn3 = nn.BatchNorm1d(120)
        
        # F6: Fully connected to 84 units (used for 7x12 bitmap output)
        self.fc2 = nn.Linear(120, 84)

        # F7: New output layer which will give softmax activations for each of the 10 digits
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.bn1(F.relu(self.C1(x)))  # C1
        x = self.S2(x)                    # S2
        x = self.bn2(F.relu(self.C3(x)))  # C3
        x = self.S4(x)                    # S4
        x = x.view(-1, 16 * 5 * 5)        # Flatten
        x = self.dropout(self.bn3(F.relu(self.fc1(x)))) # C5
        x = self.dropout(F.relu(self.fc2(x)))           # F6
        x = self.fc3(x)  # F7
        return x

def train(model, train_loader, test_loader, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    error_rates=[]
    test_error_rates=[]
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        total, correct = 0, 0
        for image, label in train_loader:
            model.zero_grad()
            output = model(image)
            loss = F.cross_entropy(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(output, dim=1)      # Make prediction for all digits in the batch
            correct += (preds == label).sum().item() # Count how many correct in batch
            total += label.size(0)                   # Total samples this batch

        error_rate = 1-(correct/total)
        error_rates.append(error_rate)
        total, correct = 0, 0
        # Seperately track the error rate each epoch on test set without backwards propagation 
        for image, label in test_loader:
            output = model(image)
            preds = torch.argmax(output, dim=1)      
            correct += (preds == label).sum().item() 
            total += label.size(0)                   
        error_rate = 1-(correct/total)
        test_error_rates.append(error_rate)
    return error_rates, test_error_rates

def main():

    train_affine = torchvision.transforms.Compose([
        torchvision.transforms.Pad(2, fill=0, padding_mode='constant'), 
        torchvision.transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.7, 1)),
        torchvision.transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    ])

    affine = torchvision.transforms.Compose([
        torchvision.transforms.Pad(2, fill=0, padding_mode='constant'), 
        torchvision.transforms.RandomAffine(degrees=10, translate=(0.3, 0.3), scale=(0.7, 1)),
        torchvision.transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    ])
    train_set = mnist.MNIST(split="train",transform=affine)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_set = mnist.MNIST(split="test",transform=affine)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    model = LeNetMod()
    error_rates, test_error_rates = train(model, train_loader, test_loader)
    torch.save(model, "LeNet2.pth")

    # Conversion to percentages
    error_rates = [e * 100 for e in error_rates]
    test_error_rates = [e * 100 for e in test_error_rates]

    print(f'Train error rate at epoch 20 = {error_rates[19]}%')
    print(f'Test error rate at epoch 20 = {test_error_rates[19]}%')

    x = (list(range(1,21))) # x-ticks for 20 epochs
    plt.plot(x, error_rates, marker="o", label="Train Error")
    plt.plot(x, test_error_rates, marker="s", label="Test Error")
    plt.xticks(x)
    plt.xlabel("Epoch")
    plt.ylabel("Error Rate (%)")
    plt.title("Training vs Test Error")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("error_plot2.png")
    plt.show()

if __name__=="__main__":

    main()