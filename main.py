"""
MNIST Digit Recognition - Main Pipeline (PyTorch Version)
"""

import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import MNISTDataLoader
from visualization import MNISTVisualizer
from cnn_model import MNISTClassifier
import numpy as np
from sklearn.model_selection import train_test_split

def run_eda():
    """Run exploratory data analysis"""
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    loader = MNISTDataLoader()
    (X_train, y_train), (X_test, y_test) = loader.load_data()
    
    # Visualize samples
    MNISTVisualizer.plot_digit_samples(X_train, y_train, n_samples=25)
    
    # Plot class distribution
    MNISTVisualizer.plot_class_distribution(y_train, y_test)
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Image shape: {X_train[0].shape}")
    print(f"Unique labels: {np.unique(y_train)}")
    
    return (X_train, y_train), (X_test, y_test)

def run_cnn(X_train, y_train, X_test, y_test, epochs=5):
    """Run CNN training and evaluation"""
    print("=" * 60)
    print("CONVOLUTIONAL NEURAL NETWORK (PyTorch)")
    print("=" * 60)
    
    # Split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )
    
    # Build and train CNN
    classifier = MNISTClassifier()
    
    print("\nTraining CNN...")
    history = classifier.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=64
    )
    
    # Evaluate on test set
    print("\nEvaluating CNN...")
    test_loss, test_acc = classifier.evaluate(X_test, y_test)
    
    # Make predictions
    y_pred = classifier.predict(X_test[:100])  # Predict on first 100 samples
    
    # Plot confusion matrix for first 100 samples
    MNISTVisualizer.plot_confusion_matrix(
        y_test[:100], y_pred,
        classes=range(10),
        figsize=(10, 8)
    )
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    classifier.save('models/cnn_model.pth')
    
    return test_acc

def run_all(epochs=5):
    """Run complete pipeline"""
    print("=" * 60)
    print("MNIST DIGIT RECOGNITION - COMPLETE PIPELINE")
    print("=" * 60)
    
    # Step 1: EDA
    (X_train, y_train), (X_test, y_test) = run_eda()
    
    # Step 2: CNN
    accuracy = run_cnn(X_train, y_train, X_test, y_test, epochs=epochs)
    
    print("=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"CNN Test Accuracy: {accuracy:.2f}%")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='MNIST Digit Recognition Pipeline')
    parser.add_argument('--eda', action='store_true', help='Run EDA only')
    parser.add_argument('--cnn', action='store_true', help='Run CNN only')
    parser.add_argument('--all', action='store_true', help='Run complete pipeline')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    
    args = parser.parse_args()
    
    if args.eda:
        run_eda()
    elif args.cnn:
        # Load data first
        loader = MNISTDataLoader()
        (X_train, y_train), (X_test, y_test) = loader.load_data()
        run_cnn(X_train, y_train, X_test, y_test, epochs=args.epochs)
    elif args.all:
        run_all(epochs=args.epochs)
    else:
        # Default: run everything
        run_all(epochs=args.epochs)

if __name__ == "__main__":
    main()
