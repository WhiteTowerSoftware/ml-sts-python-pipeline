"""This will be used as an entry point when serving the model"""
from sagemaker_containers.beta.framework import content_types, encoders
import numpy as np
import joblib
import os


def model_fn(model_dir):
    """Deserialized and return fitted model
    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf


def input_fn(input_data: str, content_type: str) -> np.array:
    """Process the endpoint request"""
    np_array = encoders.decode(input_data, content_type)
    if content_type in content_types.UTF8_TYPES:
        ret = np_array.astype(np.float32)
    else:
        ret = np_array

    if len(ret.shape) == 1:
        # the model expect a 2D array
        ret = ret.reshape(1,-1)

    return ret
