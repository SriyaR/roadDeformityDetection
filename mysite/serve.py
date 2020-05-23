import tensorflow as tf
import numpy as np

def serve_ssd_model():
    TFLITE_MODEL = "/home/sriyaR/mysite/detect.tflite"

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

    TFLITE_MODEL = "/home/sriyaR/mysite/UNet_25_Crack.tflite"

    tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)

    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()
    tflite_interpreter.allocate_tensors()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    return tflite_interpreter, height, width, input_details, output_details

def serve_rcnn_model():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.gfile.GFile("/home/sriyaR/mysite/frozen_inference_graph.pb", 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph
