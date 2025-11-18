from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import mnist
import torch
import numpy as np
import torchvision
from lenet1 import LeNet5

def test(dataloader,model):

    rbf_targets = torch.load("rbf_targets.pt")
    total, correct = 0,0
    truth = []
    preds = []
    confident_misclassifications = {}
    for image, label in dataloader:
        label = label.item()  
        output = model(image)
        distances = ((output[0] - rbf_targets) ** 2).sum(dim=1)
        pred = torch.argmin(distances) # Find prediction as smallest euclidian distance to rbf
        confidence = -distances[pred] # Lower distance -> higher confidence
        if pred==label: correct+=1
        elif label not in confident_misclassifications or confidence > confident_misclassifications[label][0]:
            confident_misclassifications[label] = [confidence, image.squeeze(0), pred.item()]
        total+=1
        truth.append(label)
        preds.append(pred.item())

    print(f'Model accuracy = {(correct/total)*100}%')
    cm = confusion_matrix(truth, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()
    plt.close()

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # 2 rows × 5 columns
    for digit in range(10):
        ax = axes[digit // 5][digit % 5]
        if digit in confident_misclassifications:
            _, image, prediction = confident_misclassifications[digit]
            img_np = image.squeeze(0).numpy()  # Convert to [32, 32]
            ax.imshow(img_np, cmap='gray')
            ax.set_title(f"{digit}\nPrediction: {prediction}")
        else:
            ax.set_title(f"No mistake found for {digit}")
        ax.axis("off")

    plt.suptitle("Most Confident Digit Misclassifications", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("most_confident_misclassifications.png")
    plt.show()

def main():

    pad=torchvision.transforms.Pad(2,fill=0,padding_mode='constant')

    mnist_test=mnist.MNIST(split="test",transform=pad)

    test_dataloader= DataLoader(mnist_test,batch_size=1,shuffle=False)

    model = torch.load("LeNet1.pth")

    test(test_dataloader,model)

 

if __name__=="__main__":

    main()