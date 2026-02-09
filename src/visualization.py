"""
Visualization utilities for MNIST
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

class MNISTVisualizer:
    @staticmethod
    def plot_digit_samples(X, y, n_samples=25, figsize=(10, 10)):
        """Plot sample digits with their labels"""
        fig, axes = plt.subplots(5, 5, figsize=figsize)
        axes = axes.ravel()
        
        for i in range(n_samples):
            axes[i].imshow(X[i].reshape(28, 28), cmap='gray')
            axes[i].set_title(f"Label: {y[i]}")
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_class_distribution(y_train, y_test):
        """Plot class distribution for train and test sets"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Train distribution
        train_unique, train_counts = np.unique(y_train, return_counts=True)
        axes[0].bar(train_unique, train_counts)
        axes[0].set_title('Training Set Class Distribution')
        axes[0].set_xlabel('Digit')
        axes[0].set_ylabel('Count')
        
        # Test distribution
        test_unique, test_counts = np.unique(y_test, return_counts=True)
        axes[1].bar(test_unique, test_counts)
        axes[1].set_title('Test Set Class Distribution')
        axes[1].set_xlabel('Digit')
        axes[1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 8)):
        """Plot confusion matrix"""
        if classes is None:
            classes = range(10)
            
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return cm
    
    @staticmethod
    def plot_training_history(history, figsize=(12, 4)):
        """Plot training history (accuracy and loss)"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Accuracy plot
        axes[0].plot(history.history['accuracy'], label='Train Accuracy')
        if 'val_accuracy' in history.history:
            axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1].plot(history.history['loss'], label='Train Loss')
        if 'val_loss' in history.history:
            axes[1].plot(history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    print("MNIST Visualizer module loaded successfully")
