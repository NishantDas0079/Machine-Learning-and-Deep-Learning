"""
MNIST Data Loader - PyTorch Version
"""

import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch

class MNISTDataLoader:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
    def load_data(self, normalize=True):
        """Load MNIST dataset using PyTorch"""
        print("Loading MNIST dataset with PyTorch...")
        
        # Define transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load training data
        train_dataset = datasets.MNIST(
            root='./data', 
            train=True, 
            download=True, 
            transform=transform
        )
        
        # Load test data
        test_dataset = datasets.MNIST(
            root='./data', 
            train=False, 
            download=True, 
            transform=transform
        )
        
        # Convert to numpy arrays
        self.X_train = train_dataset.data.numpy()
        self.y_train = train_dataset.targets.numpy()
        self.X_test = test_dataset.data.numpy()
        self.y_test = test_dataset.targets.numpy()
        
        # Normalize to [0, 1] if requested
        if normalize:
            self.X_train = self.X_train.astype('float32') / 255.0
            self.X_test = self.X_test.astype('float32') / 255.0
            
        # Reshape for CNN (add channel dimension)
        self.X_train = self.X_train.reshape(-1, 28, 28, 1)
        self.X_test = self.X_test.reshape(-1, 28, 28, 1)
        
        print(f"Training data shape: {self.X_train.shape}")
        print(f"Test data shape: {self.X_test.shape}")
        
        return (self.X_train, self.y_train), (self.X_test, self.y_test)
    
    def visualize_samples(self, n_samples=25):
        """Visualize sample digits"""
        if self.X_train is None:
            print("Please load data first")
            return
            
        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        axes = axes.ravel()
        
        for i in range(n_samples):
            # Remove channel dimension for display
            if len(self.X_train[i].shape) == 3:
                img = self.X_train[i].reshape(28, 28)
            else:
                img = self.X_train[i]
                
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"Label: {self.y_train[i]}")
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.show()
    
    def get_class_distribution(self):
        """Get distribution of classes"""
        if self.y_train is None:
            print("Please load data first")
            return
            
        unique, counts = np.unique(self.y_train, return_counts=True)
        return dict(zip(unique, counts))

# Test the data loader
if __name__ == "__main__":
    loader = MNISTDataLoader()
    (X_train, y_train), (X_test, y_test) = loader.load_data()
    print(f"Class distribution: {loader.get_class_distribution()}")
    loader.visualize_samples()
