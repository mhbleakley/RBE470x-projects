import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define the BombermanModel class, which is a subclass of nn.Module
class BombermanModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define three fully connected layers
        self.fc1 = nn.Linear(16, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)

    def forward(self, x):
        # Apply ReLU activation function to output of each layer
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        # Final layer has no activation function
        x = self.fc3(x)
        return x

# Define the BombermanDataset class, which is a subclass of Dataset
class BombermanDataset(Dataset):
    def __init__(self, data):
        # Initialize dataset with input data
        self.data = data

    def __getitem__(self, index):
        # Extract input data and target label from dataset at given index
        x = torch.FloatTensor(self.data[index][:16])
        y = torch.LongTensor([self.data[index][-1]])
        return x, y

    def __len__(self):
        # Return length of dataset
        return len(self.data)

# Define the train function to train the model
def train(model, train_loader, optimizer, criterion):
    # Set model to training mode
    model.train()
    running_loss = 0.0
    # Iterate over batches in the training data
    for inputs, labels in train_loader:
        # Reset optimizer gradients
        optimizer.zero_grad()
        # Forward pass to compute model outputs
        outputs = model(inputs)
        # Compute loss between model outputs and target labels
        loss = criterion(outputs, labels.squeeze())
        # Backward pass to compute gradients
        loss.backward()
        # Update model parameters
        optimizer.step()
        # Accumulate running loss
        running_loss += loss.item()
    # Compute average training loss
    return running_loss / len(train_loader)

# Define the test function to evaluate the model
def test(model, test_loader, criterion):
    # Set model to evaluation mode
    model.eval()
    running_loss = 0.0
    total_correct = 0
    # Iterate over batches in the test data
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Forward pass to compute model outputs
            outputs = model(inputs)
            # Compute loss between model outputs and target labels
            loss = criterion(outputs, labels.squeeze())
            # Accumulate running loss
            running_loss += loss.item()
            # Count number of correct predictions
            total_correct += (outputs.argmax(dim=1) == labels.squeeze()).sum().item()
    # Compute average test loss and accuracy
    return running_loss / len(test_loader), total_correct / len(test_loader.dataset)

# Define input data
data = [[1, 1, 1, 1, True, False, False, False, True, 1],
[1, 1, 1, 1, True, False, False, False, True, 2],
[1, 1, 1, 1, True, False, False, False, True, 0],
[1, 1, 1, 1, True, False, False, False, False, 5],
[0, 1, 1, 1, True, False, False, False, True, 2],
[0, 1, 1, 1, True, False, False, False, True, 3],
[0, 1, 1, 1, True, False, False, False, False, 1],
[0, 0, 1, 1, True, False, False, False, True, 3],
[0, 0, 1, 1, True, False, False, False, False, 0]]

# Split data into training and test sets
train_data = data[:6]
test_data = data[6:]

# Create instances of the BombermanDataset class for training and test sets
train_dataset = BombermanDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_dataset = BombermanDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# Create instance of the BombermanModel class
model = BombermanModel()

# Define optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Train the model for 10 epochs
for epoch in range(10):
    # Train the model on the training set
    train_loss = train(model, train_loader, optimizer, criterion)
    # Evaluate the model on the test set
    test_loss, test_accuracy = test(model, test_loader, criterion)
    # Print training and test loss and accuracy for each epoch
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, Test Accuracy = {test_accuracy:.4f}")