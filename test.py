import json
import torch
import torch.nn as nn
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import Dataset, DataLoader
import random
from torch.nn.functional import softmax

# Load intents
with open('description.json', 'r') as file:
    intents = json.load(file)

# Create vectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform([pattern for intent in intents['intents'] for pattern in intent['patterns']])

# Create dataset
class ChatDataset(Dataset):
    def __init__(self, intents, vectorizer):
        self.vectorizer = vectorizer
        self.data = []
        for intent in intents['intents']:
            for pattern in intent['patterns']:
                self.data.append({
                    'input': self.vectorizer.transform([pattern]).toarray()[0],
                    'output': intents['intents'].index(intent),
                })
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

dataset = ChatDataset(intents, vectorizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Create model
class ChatModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChatModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = ChatModel(X.shape[1], len(intents['intents']))
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Train model
for epoch in range(500):
    for data in dataloader:
        input_data = torch.tensor(data['input'].float())
        output_data = torch.tensor(data['output'])
        optimizer.zero_grad()
        predictions = model(input_data)
        loss = loss_function(predictions, output_data)
        loss.backward()
        optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()}')


# Save model
torch.save(model.state_dict(), 'chatpulse_response.pth')

# Test model
while True:
    message = input('You: ')
    if message == 'quit':
        break
    with torch.no_grad():
        prediction = model(torch.tensor(vectorizer.transform([message]).toarray()[0]).float())
        prediction = softmax(prediction, dim=0)  # Apply softmax to the outputs
        confidence, predicted_idx = torch.max(prediction, 0)
        print('Confidence:', confidence.item())  # print the confidence score
        if confidence < 0.7:  # confidence threshold
            print('ChatBot: I did not understand that. Could you please rephrase?')
        else:
            tag = intents['intents'][predicted_idx.item()]['tag']
            response = random.choice([intent['responses'] for intent in intents['intents'] if intent['tag'] == tag][0])
            print('ChatBot:', response,confidence)