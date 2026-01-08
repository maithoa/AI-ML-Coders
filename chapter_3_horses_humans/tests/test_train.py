import pytest
import numpy as np
import tensorflow as tf

from src.model_factory import create_model 
from src.train import train_model

@pytest.fixture
def dummy_data():
    """
    Creates a small set of random images and labels for testing.
    """
    # Create 64 random images of 300x300x3
    x_train = np.random.random((64, 300, 300, 3)).astype(np.float32)
    # Create 64 random binary labels (0 or 1)
    y_train = np.random.randint(2, size=(64, 1)).astype(np.float32)
    
    return x_train, y_train

def test_train_model_runs_without_error(dummy_data):
    """
    Integration test: Ensures that the model can complete at least 1 epoch 
    of training without crashing.
    """
    x_train, y_train = dummy_data
    # Wrap numpy arrays into a tf.data.Dataset 
    # This mimics the behavior of a DirectoryIterator/Generator
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
    val_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

    model = create_model(input_shape_param=(300, 300, 3))
    
    # Minimal hyperparams for a quick test
    hyperparams = {
        'learning_rate': 0.001,
        'epochs': 1,
        'steps_per_epoch': 2,
        'val_steps': 1
    }

    
    # We use the raw numpy data as a simple generator-like source
    # for testing the training logic
    history = train_model(
        model=model,
        train_data=train_ds, # Testing with direct data
        val_data=val_ds,
        hyperparams=hyperparams
    )
    
    # Assertions
    assert history is not None
    assert 'accuracy' in history.history
    assert len(history.history['loss']) == 1 # Check if 1 epoch completed