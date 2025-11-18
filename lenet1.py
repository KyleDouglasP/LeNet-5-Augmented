import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import mnist
import matplotlib.pyplot as plt

'''

Steps I used from paper
1. Input
The input is a 32 * 32 pixel image.

2. C1 Layer  First Convolutional Layer
The first layer is a convolutional layer (C1) with six feature maps.
Each unit in each feature map is connected to a 5 * 5 receptive field in the input. 
The size of each feature map is 28 * 28. The kernel size is 5 * 5. The activation function is a scaled tanh: y = 1.7159 tanh(2x/3).

3. S2 Layer  Subsampling Layer
Layer S2 is a subsampling layer with six feature maps of size 14 * 14. 
Each unit computes the average of a 2 * 2 neighborhood in the corresponding feature map in C1, multiplies it by a trainable coefficient (different for each map), adds a trainable bias, and applies the scaled tanh function.

4. C3 Layer  Second Convolutional Layer
Layer C3 is a convolutional layer with 16 feature maps of size 10 * 10. 
Each unit is connected to several (but not all) S2 maps through 5 * 5 kernels. 

5. S4 Layer  Subsampling
Layer S4 is a subsampling layer with 16 feature maps of size 5 * 5. 
Each unit performs a similar operation as in S2, i.e., averaging over 2 * 2 regions with trainable coefficients and biases followed by the scaled tanh activation function.

6. C5 Layer Fully Connected Convolution
Layer C5 is a convolutional layer with 120 feature maps of size 1 * 1. 
Since the size of S4s feature maps is also 5 * 5, the 5 * 5 kernels cover the entire input. 

7. F6 Layer  Fully Connected to 84 Units
Layer F6 is a fully connected layer with 84 units. 
The activation function is again the scaled tanh. 

8. Output Layer (Used for Bitmap Comparison)
The output layer contains as many units as there are classes to be recognized. In our case, the output of the network is not a class index, but a bitmap image of size 7 × 12 pixels, representing the digit.

'''

def scaled_tanh(x):
    #y = 1.7159 * tanh(2x/3)
    return 1.7159 * torch.tanh((2.0 / 3.0) * x)

# Implementation of equation 9 in the paper
def loss9(output, label, rbf_targets, j=0.1):
    distances = torch.stack([((output-t)**2).sum() for t in rbf_targets])
    correct = distances[label]
    # Remove correct class from distances
    incorrect_distances = torch.cat((distances[:label], distances[label+1:]))
    j_tensor = torch.tensor(j, device=incorrect_distances.device)
    log = torch.log(torch.exp(-j_tensor) + torch.exp(-incorrect_distances).sum())
    return correct + log


class LeNet5(nn.Module):

    def __init__(self, input_channels=1):
        super(LeNet5, self).__init__()

        # C1: Conv Layer (input: 32x32 → output: 28x28)
        self.C1 = nn.Conv2d(in_channels=input_channels, out_channels=6, kernel_size=5)
        
        # S2: Avg Pool (28x28 → 14x14)
        self.S2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # C3: Conv Layer (14x14 → 10x10)
        self.C3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        
        # S4: Avg Pool (10x10 → 5x5)
        self.S4 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # C5: Fully connected convolution (5x5 input maps × 16 channels = 400)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        
        # F6: Fully connected to 84 units (used for 7x12 bitmap output)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x):
        x = scaled_tanh(self.C1(x))  # C1
        x = self.S2(x)               # S2
        x = scaled_tanh(self.C3(x))  # C3
        x = self.S4(x)               # S4
        x = x.view(-1, 16 * 5 * 5)   # Flatten (C5)
        x = scaled_tanh(self.fc1(x)) # C5 → FC1
        x = scaled_tanh(self.fc2(x)) # F6
        return x

def train(model, rbf_targets, train_loader, test_loader, epochs=20, step_size=0.001):
    error_rates=[]
    test_error_rates=[]
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        total, correct = 0, 0
        for image, label in train_loader:
            label = label.item()  
            output = model(image)
            loss = loss9(output[0], label, rbf_targets)
            model.zero_grad()
            loss.backward()
            # Performs steepest gradient method
            with torch.no_grad():
                for param in model.parameters():
                    param -= step_size * param.grad
            pred = torch.argmin(((output[0] - rbf_targets) ** 2).sum(dim=1)) # Find prediction as smallest euclidian distance to rbf
            if pred==label: correct+=1
            total+=1
        error_rate = 1-(correct/total)
        error_rates.append(error_rate)
        total, correct = 0, 0
        # Seperately track the error rate each epoch on test set without backwards propagation 
        for image, label in test_loader:
            label = label.item()  
            output = model(image)
            pred = torch.argmin(((output[0] - rbf_targets) ** 2).sum(dim=1)) # Find prediction as smallest euclidian distance to rbf
            if pred==label: correct+=1
            total+=1
        error_rate = 1-(correct/total)
        test_error_rates.append(error_rate)
    return error_rates, test_error_rates

def main():

    rbf_targets = torch.load("rbf_targets.pt")

    pad=torchvision.transforms.Pad(2,fill=0,padding_mode='constant')
    train_set = mnist.MNIST(split="train",transform=pad)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    test_set = mnist.MNIST(split="test",transform=pad)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

    model = LeNet5()
    error_rates, test_error_rates = train(model, rbf_targets, train_loader, test_loader)
    torch.save(model, "LeNet1.pth")

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
    plt.savefig("error_plot.png")
    plt.show()

if __name__=="__main__":

    main()