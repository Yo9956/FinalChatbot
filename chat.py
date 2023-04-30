import random
import json

import torch

from model import NeuralNet
from nltkFile import bag_of_words, tokenize

device = torch.device('cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

inputSize = data["inputSize"]
hidSize = data["hidSize"]
outputSize = data["outputSize"]
allWords = data['allWords']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(inputSize, hidSize, outputSize).to(device)
model.load_state_dict(model_state)
model.eval()


chatBotName = "QUBot"

def getResponse(message):
    sentence = tokenize(message)
    X = bag_of_words(sentence, allWords)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    print(tag)
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "the system is still under development......"