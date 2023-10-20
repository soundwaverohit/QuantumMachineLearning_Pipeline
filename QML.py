## Imports 
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Function
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

import qiskit
from qiskit import transpile, assemble
from qiskit.visualization import * 

import matplotlib.pyplot as plt

from Net import Net
from QuantumCircuit import QuantumCircuit
from HybridFunction import HybridFunction
from Hybrid import Hybrid

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Parameter - for training data
n_samples = 100

xTrain = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

idx = np.append(np.where(xTrain.targets == 0)[0][:n_samples], 
                np.where(xTrain.targets == 1)[0][:n_samples])

xTrain.data = xTrain.data[idx]
xTrain.targets = xTrain.targets[idx]

train_loader = torch.utils.data.DataLoader(xTrain, batch_size=1, shuffle=True)

# Parameter = for testing data
n_samples = 50

xTest = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

idx = np.append(np.where(xTest.targets == 0)[0][:n_samples], 
                np.where(xTest.targets == 1)[0][:n_samples])

xTest.data = xTest.data[idx]
xTest.targets = xTest.targets[idx]

test_loader = torch.utils.data.DataLoader(xTest, batch_size=1, shuffle=True)


model = Net()
model.to(device)

optimizer = optim.Adam(model.parameters(), lr = 1e-3)
loss_fn = nn.NLLLoss()

epochs = 5
list_loss = []

model.train()
for epoch in range(epochs):
    tl = []
    for batch, (data, target) in enumerate(train_loader):
        
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        
        tl.append(loss.item())
        
    list_loss.append(sum(tl)/len(tl))
    print(f'Training {epoch+1}/{epochs} Loss : {abs(list_loss[-1])}')


plt.plot(list_loss)

plt.title('Hybrid Neural Network Training Covergence')

plt.xlabel('Training Iterations')
plt.ylabel('Negative Log Likelihood Loss')
plt.savefig('Fig')

print()
print("Evaluation Loop")

model.eval()

with torch.no_grad():
    correct = 0
    for batch, (data, target) in enumerate(test_loader):
        output = model(data)
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        loss = loss_fn(output, target)
        tl.append(loss.item())
        
    print(f'Accuracy: {correct/len(test_loader) * 100}')