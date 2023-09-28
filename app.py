from flask import Flask, request, render_template, jsonify
import os
import tensorflow as tf
from werkzeug.utils import secure_filename
from classifier import classify


app = Flask(__name__)
STATIC_FOLDER = "static"
UPLOAD_FOLDER = "static/uploads/"

if not os.path.exists(UPLOAD_FOLDER):
   os.makedirs(UPLOAD_FOLDER)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

cnn_model = tf.keras.models.load_model(os.path.join(STATIC_FOLDER, "models", "sport_balls_model.h5"))

# Function to check if the file extension is allowed
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def home():
    return render_template("base.html")


@app.route("/classify", methods=["POST"])
def classify_image():
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        # Securely generate a filename and save the file
        filename = secure_filename(file.filename)
        upload_image_file = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_image_file)

        # Classify the uploaded image
        result, probability = classify(cnn_model, upload_image_file)
        probability = probability[:5]

        # Check if the result is a tuple
        if isinstance(result, tuple) and len(result) > 0:
            label = result[0]
        else:
            label = result

        if isinstance(label, tuple):
            label = list(label)
        return render_template("result.html", label=label, probability=probability)
    else:
        return jsonify({"error": "Invalid file format"}), 400


if __name__ == "__main__":
    app.run(debug=True)
