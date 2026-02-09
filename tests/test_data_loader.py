"""
Test data loader
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from data_loader import MNISTDataLoader
import numpy as np

def test_data_loading():
    """Test if data loads correctly"""
    loader = MNISTDataLoader()
    (X_train, y_train), (X_test, y_test) = loader.load_data()
    
    # Check shapes
    assert X_train.shape == (60000, 28, 28, 1), f"Expected (60000, 28, 28, 1), got {X_train.shape}"
    assert X_test.shape == (10000, 28, 28, 1), f"Expected (10000, 28, 28, 1), got {X_test.shape}"
    
    # Check labels
    assert len(np.unique(y_train)) == 10, "Should have 10 unique labels"
    assert len(np.unique(y_test)) == 10, "Should have 10 unique labels"
    
    # Check normalization
    assert X_train.max() <= 1.0, "Data should be normalized to [0, 1]"
    assert X_train.min() >= 0.0, "Data should be normalized to [0, 1]"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_data_loading()
