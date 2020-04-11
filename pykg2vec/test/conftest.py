import pytest
import tensorflow as tf

@pytest.fixture(scope="session", autouse=True)
def switch_on_eager_execution(request):
    """Setup eager execution within the pytest runtime for better visibility to Coverage.py"""
    tf.config.experimental_run_functions_eagerly(True)