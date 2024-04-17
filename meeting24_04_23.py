# Computational Materials L&D Team MILL Georgia Tech
# Guilherme Ryuji Weber Nakamura
# Materials Classification Database: Ionic, Covalent, Metallic
# Version 1.0 - April 2024

import copy
import tqdm
import torch 
from torch import nn
import numpy as np
import sklearn as skl
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder
from numpy import reshape
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

#DATA EXTRACTION AND FORMATTING

#Data Loading
data = pd.read_csv('MaterialClassifcation.csv')
head = data.head()
X = data.iloc[:,2:]
y = data.iloc[:,1]
y = y.to_numpy()
y = y.reshape(-1,1)

#One Hot Encoding for Multiclass
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y)
y = ohe.transform(y)

#Data Splitting
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, shuffle=True)

#MODEL

class NeuralMatCat(nn.Module):
    
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()
        self.hidden = nn.Linear(11, 5)
        self.act = nn.ReLU()
        self.output = nn.Linear(5, 3)
        
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x

model = NeuralMatCat().to(device)

#PARAMETERS
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

n_epochs = 20
batch_size = 5
batches_per_epoch = len(X_train) 

best_acc = - np.inf   # init to negative infinity
best_weights = None
train_loss_hist = []
train_acc_hist = []
test_loss_hist = []
test_acc_hist = []
 
#TRAINING
for epoch in range(n_epochs):
    epoch_loss = []
    epoch_acc = []
    # set model in training mode and run through each batch
    model.train()
    with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for i in bar:
            # take a batch
            start = i * batch_size
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # compute and store metrics
            acc = (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()
            epoch_loss.append(float(loss))
            epoch_acc.append(float(acc))
            bar.set_postfix(
                loss=float(loss),
                acc=float(acc)
            )
            
    #TESTING
    model.eval()
    y_pred = model(X_test)
    ce = loss_fn(y_pred, y_test)
    acc = (torch.argmax(y_pred, 1) == torch.argmax(y_test, 1)).float().mean()
    ce = float(ce)
    acc = float(acc)
    train_loss_hist.append(np.mean(epoch_loss))
    train_acc_hist.append(np.mean(epoch_acc))
    test_loss_hist.append(ce)
    test_acc_hist.append(acc)
    if acc > best_acc:
        best_acc = acc
        best_weights = copy.deepcopy(model.state_dict())
    print(f"Epoch {epoch} validation: Cross-entropy={ce}, Accuracy={acc}")
    
#PLOTTING 
plt.plot(train_loss_hist, label="train")
plt.plot(test_loss_hist, label="test")
plt.xlabel("epochs")
plt.ylabel("cross entropy")
plt.legend()
plt.show()
 
plt.plot(train_acc_hist, label="train")
plt.plot(test_acc_hist, label="test")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()

#BEST MODEL
model.load_state_dict(best_weights)

#Examples
#predictions = model(X[9])
#print(predictions)
#9 Silver,Metal,83,195.93,201.24,304.425,18.6059,235,10490,1105.88,2162,63,624
#25 Alumina,Ceramic,380,310,380,30,6.5,880,3960,2054,4377,100,9
#42 PE,Polymer,1.41421,14.1421,21.2132,0.447214,141.421,2100,950,1243.22,4377,1.41421,1.8

#predicted_labels = np.argmax(predictions, axis=1)
#print("Predicted Labels:", predicted_labels)

# #SAVE MODEL
# torch.save(model.state_dict(), "model.pth")
# print("Saved PyTorch Model State to model.pth")

# #LOAD MODEL
# model = NeuralMatCat().to(device)
# model.load_state_dict(torch.load("model.pth"))