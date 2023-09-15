from flask import Flask, request
import os
import tensorflow as tf
from classifier import classify

app = Flask(__name__)

STATIC_FOLDER = "static"
UPLOAD_FOLDER = "static/uploads/"

cnn_model = tf.keras.models.load_model(STATIC_FOLDER + "/models/" + "!!!!!")  # TODO Write file name for the model


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.post("/classify")
def upload_file():
    file = request.files["image"]
    upload_image_file = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(upload_image_file)

    label = classify(cnn_model, upload_image_file)

    return {
        "label": label,
    }


if __name__ == '__main__':
    app.run()
