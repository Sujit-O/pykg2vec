import pytest
import tensorflow as tf

@pytest.fixture(scope="session", autouse=True)
def run_tf_function_eagerly(request):
    tf.config.experimental_run_functions_eagerly(True)

def switch_on_eager_execution():
    pass