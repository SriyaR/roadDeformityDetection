import tensorflow as tf
import numpy as np
from theanos.keras.models import load_model

def serve_ssd_model():
    #change this
    TFLITE_MODEL = "detect.tflite"

    tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)

    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()
    tflite_interpreter.allocate_tensors()
    floating_img = input_details[0]['dtype'] == np.float32
    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    return tflite_interpreter, floating_img, height, width, input_details, output_details


def serve_unet_model():

    H5_MODEL = "segment.h5"
    model = load_model(H5_MODEL)
    width = 128
    height = 128
    return model,height,width
