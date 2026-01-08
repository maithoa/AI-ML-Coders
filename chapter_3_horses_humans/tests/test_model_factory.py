import pytest
import tensorflow as tf
import keras

from src.model_factory import create_model 

# We use a fixture to initialize the model once for all test functions
@pytest.fixture
def model():
    """
    Provides a freshly initialized model for each test function.
    """
    input_shape = (300, 300, 3)
    return create_model(input_shape_param=input_shape)

def test_model_input_shape(model):
    """
    Verify the input layer matches the expected (None, 300, 300, 3).
    """
    expected_shape = (None, 300, 300, 3)
    assert model.input_shape == expected_shape

def test_model_output_shape(model):
    """
    Verify the output layer shape is (None, 1) for binary classification.
    """
    assert model.output_shape == (None, 1)

def test_last_layer_activation(model):
    """
    Verify the final activation function is sigmoid.
    """
    last_layer = model.layers[-1]
    assert last_layer.activation.__name__ == 'sigmoid'

def test_is_sequential(model):
    """
    Ensure the returned object is indeed a Keras Sequential model.
    """
    assert isinstance(model, keras.models.Sequential)