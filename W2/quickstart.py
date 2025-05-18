#!/usr/bin/env python

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
# from torchtext import datasets
from torchvision.transforms import ToTensor

training_data=datasets.FashionMNIST(root="data",
                                    train=True,
                                    download=True,
                                    transform=ToTensor())

test_data=datasets.FashionMNIST(root="data",
                                      train=True,
                                      download=True,
                                      transform=ToTensor())

print(training_data)

batch_size=64

train_dataloader=DataLoader(training_data,batch_size=batch_size)
test_dataloader=DataLoader(test_data,batch_size=batch_size)

device="cpu"

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten=nn.Flatten()
        # ReLU is the activation function
        self.linear_relu_stack=nn.Sequential(nn.Linear(784,512),
                                             nn.ReLU(),
                                             nn.Linear(512,512),
                                             nn.ReLU(),
                                             nn.Linear(512,10))
    def forward(self,x):
        x=self.flatten(x)
        logits=self.linear_relu_stack(x)
        return logits

model=NN().to(device)
print(model)

loss=nn.CrossEntropyLoss()
optim=torch.optim.SGD(model.parameters(),lr=1e-3)

def train(dataloader,model,loss,optim):
    size=len(dataloader.dataset)
    model.train()
    for i,(x,y) in enumerate(dataloader):
        x=x.to(device)
        y=y.to(device)
        pred=model(x)
        cost=loss(pred,y)
        cost.backward()
        optim.step()
        optim.zero_grad()

        if i%100==0:
            ccost,current=cost.item(),(i+1)*len(x)
            print(f"Cost: {cost:.7f} [{current:>5d}/{size:>5d}]")

def test(dataloader,model,loss):
    size=len(dataloader.dataset)
    nbatches=len(dataloader)
    model.eval()
    test_loss,correct=0,0
    with torch.no_grad():
        for x,y in dataloader:
            x=x.to(device)
            y=y.to(device)
            pred=model(x)
            test_loss+=loss(pred,y).item()
            correct+=(pred.argmax(1)==y).type(torch.float).sum().item()
        test_loss/=nbatches
        correct/=size
        print(f"Test Error:\n Accuracy: {100*correct:>0.1f}%, Avg loss: {test_loss:>0f}\n")
        
epochs=5
for i in range(epochs):
    print(f"Epoch {i+1}\n")
    train(train_dataloader,model,loss,optim)
    test(test_dataloader,model,loss)
print("Done!")
