import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet
from nltkFile import bag_of_words, tokenize, stem

with open('intents.json', 'r') as f:
    intents = json.load(f)

allWords = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        allWords.extend(w)
        xy.append((w, tag))
        

ignore_words = ['?', '.', '!']
allWords = [stem(w) for w in allWords if w not in ignore_words]
allWords = sorted(set(allWords))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(allWords), "unique stemmed words:", allWords)

X_train = []
y_train = []
for (patSentence, tag) in xy:
    bag = bag_of_words(patSentence, allWords)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)


X_train = np.array(X_train)
y_train = np.array(y_train)

number_of_epochs = 1000
batch_size = 8
learnRate = 0.001
inputSize = len(X_train[0])
hidSize = 8
outputSize = len(tags)


class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0)

device = torch.device('cpu')

model = NeuralNet(inputSize, hidSize, outputSize).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learnRate)

for epoch in range(number_of_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{number_of_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "inputSize": inputSize,
    "hidSize": hidSize,
    "outputSize": outputSize,
    "allWords": allWords,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')