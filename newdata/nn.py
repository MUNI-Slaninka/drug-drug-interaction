import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from src.utilities import create_data

X, y = create_data(folder="../dataset")
y = y.flatten().astype(int)

"""
df = pd.read_json("jaccard.json")
df.drop(["DRUG_ID1", "DRUG_ID2"], inplace=True, axis=1)

# Define features (X) and target variable (y)
X = df.drop('INTERACTION', axis=1)  # Features
y = df['INTERACTION']  # Target variable
"""

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert numpy arrays to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Define MLP architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(7, 100)  # Input layer to hidden layer
        self.fc2 = nn.Linear(100, 50)  # Hidden layer to hidden layer
        self.fc3 = nn.Linear(50, 2)    # Hidden layer to output layer
        self.relu = nn.ReLU()          # Activation function

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model, loss function, and optimizer
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
with torch.no_grad():
    model.eval()
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = matthews_corrcoef(y_test.numpy(), predicted.numpy())
    print("Matthews Correlation Coefficient:", accuracy)