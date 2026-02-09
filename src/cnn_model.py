"""
Simple CNN Model for MNIST - PyTorch Version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # After 3 pools: 28/2/2/2 = 3.5 -> 3
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Input shape: (batch_size, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))  # -> (batch_size, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # -> (batch_size, 64, 7, 7)
        x = self.pool(F.relu(self.conv3(x)))  # -> (batch_size, 128, 3, 3)
        
        # Flatten
        x = x.view(-1, 128 * 3 * 3)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class MNISTClassifier:
    def __init__(self, device=None):
        self.model = SimpleCNN()
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training history
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=10, batch_size=32):
        """Train the model"""
        
        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        y_train_tensor = torch.LongTensor(y_train)
        
        # Create DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Validation data if provided
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).permute(0, 3, 1, 2)
            y_val_tensor = torch.LongTensor(y_val)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Training on {self.device}...")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Zero the gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
            
            train_loss = train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validation phase
            if X_val is not None and y_val is not None:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = self.model(data)
                        val_loss += self.criterion(output, target).item()
                        _, predicted = torch.max(output.data, 1)
                        val_total += target.size(0)
                        val_correct += (predicted == target).sum().item()
                
                val_loss = val_loss / len(val_loader)
                val_acc = 100 * val_correct / val_total
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            else:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        return self.history
    
    def evaluate(self, X_test, y_test, batch_size=32):
        """Evaluate the model"""
        self.model.eval()
        
        # Convert to tensors
        X_test_tensor = torch.FloatTensor(X_test).permute(0, 3, 1, 2)
        y_test_tensor = torch.LongTensor(y_test)
        
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()
        
        test_loss = test_loss / len(test_loader)
        test_acc = 100 * test_correct / test_total
        
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
        
        return test_loss, test_acc
    
    def predict(self, X):
        """Make predictions"""
        self.model.eval()
        
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        
        if len(X.shape) == 3:  # Single image (H, W, C)
            X = X.unsqueeze(0)  # Add batch dimension
        elif len(X.shape) == 4:  # Batch of images (N, H, W, C)
            X = X.permute(0, 3, 1, 2)  # Convert to (N, C, H, W)
        
        X = X.to(self.device)
        
        with torch.no_grad():
            output = self.model(X)
            _, predicted = torch.max(output.data, 1)
        
        return predicted.cpu().numpy()
    
    def save(self, filepath):
        """Save the model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load a saved model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Model loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    # Test the CNN model
    classifier = MNISTClassifier()
    print("CNN Model created successfully!")
    print(f"Using device: {classifier.device}")
    
    # Print model architecture
    print("\nModel Architecture:")
    print(classifier.model)
