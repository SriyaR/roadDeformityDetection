import flask
from flask import Flask
import numpy as np
import io
from PIL import Image, ImageDraw

from serve import serve_ssd_model
from serve import serve_unet_model


app = Flask(__name__)


def load_ssd_model():
    global tflite_interpreter, floating_img, height, width, input_details, output_details
    tflite_interpreter, floating_img, height, width, input_details, output_details = serve_ssd_model()

def load_unet_model():
    global model, height_c, width_c
    model, height_c, width_c = serve_unet_model()

load_ssd_model()
load_unet_model()

def prepare_img(image, type):
    if type == "detect":
        return Image.open(image).resize((width, height))
    elif type == "segment":
        return Image.open(image).resize((width_c, height_c))

@app.route("/detect", methods=["POST"])
def detect():
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            img = prepare_img(flask.request.files["image"], "detect")
            input_data = np.expand_dims(img, axis=0)
            if floating_img:
                input_data = (np.float32(input_data) - 128) / 128

            tflite_interpreter.set_tensor(input_details[0]['index'], input_data)
            tflite_interpreter.invoke()
            output_data = tflite_interpreter.get_tensor(output_details[0]['index'])
            output_scores = tflite_interpreter.get_tensor(output_details[2]['index'])
            results = np.squeeze(output_data)
            scores = np.squeeze(output_scores)
            size = img.size
            for r,s in zip(results, scores):
                if s>0.5:
                    draw = ImageDraw.Draw(img)
                    ymin, xmin, ymax, xmax = r
                    xmin = int(xmin * size[0])
                    xmax = int(xmax * size[0])
                    ymin = int(ymin * size[1])
                    ymax = int(ymax * size[1])
                    for x in range( 0, 4 ):
                        draw.rectangle((ymin, xmin, ymax, xmax), outline=(255, 255, 0))

    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    return rawBytes.getvalue()

@app.route("/segment", methods=["POST"])
def segment():
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            img = prepare_img(flask.request.files["image"], "segment")


            img = np.expand_dims(img, axis=0)
            img = np.float32(img)/255.0
            result = model.predict(img)
            result = result > 0.5
            result = result*255
            result = np.squeeze(result)

            img = Image.fromarray(result.astype("uint8"))
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    return rawBytes.getvalue()


@app.route('/')
def index():
    return "TARP PROJECT"

if __name__ == "__main__":
    app.run()