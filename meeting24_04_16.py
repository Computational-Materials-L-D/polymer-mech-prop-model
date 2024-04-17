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
X = data.iloc[:,3:]
y = data.iloc[:,2:]

#One Hot Encoding for Multiclass
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y)
y = ohe.transform(y)

#Data Splitting
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

#MODEL

class NeuralMatCat(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(11, 30),
            nn.ReLU(),
            nn.Linear(30, 3)
        )

model = NeuralMatCat().to(device)

#PARAMETERS
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

n_epochs = 200
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
    y_pred = model(X_test)
    ce = loss_fn(y_pred, y_test)
    acc = (torch.argmax(y_pred, 1) == torch.argmax(y_test, 1)).float().mean()
    print(f"Epoch {epoch} validation: Cross-entropy={float(ce)}, Accuracy={float(acc)}")
    
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

model.load_state_dict(best_weights)

#SAVE MODEL

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

#LOAD MODEL
model = NeuralMatCat().to(device)
model.load_state_dict(torch.load("model.pth"))

# def train(dataloader, model, loss_fn, optimizer):
#     size = len(dataloader.data)
#     model.train()
#     for batch, (X, y) in enumerate(dataloader):
#         X, y = X.to(device), y.to(device)

#         # Compute prediction error
#         pred = model(X)
#         loss = loss_fn(pred, y)

#         # Backpropagation
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

#         if batch % 100 == 0:
#             loss, current = loss.item(), (batch + 1) * len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# def test(dataloader, model, loss_fn):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     model.eval()
#     test_loss, correct = 0, 0
#     with t.torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     test_loss /= num_batches
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    

# epochs = 5
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(train_dataloader, model, loss_fn, optimizer)
#     test(test_dataloader, model, loss_fn)
# print("Done!")

#METRICS

# #Cofnusion Matrix
# r = metrics.multilabel_confusion_matrix(y_true, y_pred, labels=["White", "Black", "Red"])
# print(r)
# #Accuracy
# acc = metrics.accuracy_score(y_true, y_pred)
# print(acc)
# #Precision
# precision = metrics.precision_score(y_true, y_pred, pos_label="positive")
# print(precision)
# #Recall
# recall = metrics.recall_score(y_true, y_pred, pos_label="positive")
# print(recall)

# #F1 Score - harmonic mean of precision and recall
# f1 = metrics.f1_score(y_true, y_pred, pos_label="positive")
# #Fbeta Score - recall weigh
# fb = metrics.fbeta_score(y_true, y_pred, beta=2, pos_label="positive")
# #MCC
# #AUC
# #ROC Curve

#PLOTTING
# disp = skl.ConfusionMatrixDisplay.from_estimator(
#         classifier,
#         X_test,
#         y_test,
#         display_labels=class_names,
#         cmap=plt.cm.Blues,
#         normalize=normalize,
#     )
#     disp.ax_.set_title(title)

#     print(title)
#     print(disp.confusion_matrix)

# plt.show()

#fig = plt.figure(figsize = (10, 10))
