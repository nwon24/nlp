import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch import nn
import numpy as np

class SinData(Dataset):
    def __init__(self,file,transform=None,target_transform=None):
        self.rawdata=np.genfromtxt(file,
                                   delimiter=",",
                                   encoding="utf-8",
                                   dtype=float)
        self.transform=transform
        self.target_transform=transform
        self.ins=self.rawdata[:,0]
        self.outs=self.rawdata[:,1]
    
    def __len__(self):
        return len(self.ins)

    def __getitem__(self,idx):
        arg=torch.from_numpy(np.array([self.ins[idx]]))
        out=torch.from_numpy(np.array([self.outs[idx]]))
        #if self.transform:
            #arg=self.transform(arg)
        #if self_target_transform:
            #out=self.target_transform(out)
        return arg.to(torch.float32),out.to(torch.float32)

bsize=64
train_data=SinData("sin_train.csv",ToTensor(),ToTensor())
train_dataloader=DataLoader(train_data,batch_size=bsize,shuffle=True)
test_data=SinData("sin_test.csv",ToTensor(),ToTensor())
test_dataloader=DataLoader(test_data,batch_size=bsize,shuffle=True)

device="cpu"

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten=nn.Flatten()
        # ReLU is the activation function
        self.linear_relu_stack=nn.Sequential(nn.Linear(1,128),
                                             nn.ReLU(),
                                             nn.Linear(128,128),
                                             nn.ReLU(),
                                             nn.Linear(128,1))
    def forward(self,x):
        #x=self.flatten(x)
        logits=self.linear_relu_stack(x)
        return logits

model=NN().to(device)
loss=nn.MSELoss()
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
        
epochs=50
for i in range(epochs):
    print(f"Epoch {i+1}\n")
    train(train_dataloader,model,loss,optim)
    test(test_dataloader,model,loss)

print(model(torch.tensor([np.pi/6])))
print(model(torch.tensor([np.pi/2])))
