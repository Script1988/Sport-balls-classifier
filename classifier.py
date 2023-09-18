import tensorflow as tf
from flask import json

IMAGE_SIZE = (100, 100)


def preprocess_image(image):
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)

    return image_array


def load_and_preprocess_data(path: str):
    image = tf.keras.preprocessing.image.load_img(
        path, target_size=IMAGE_SIZE
    )

    return preprocess_image(image)


def classify(model, image_path: str):
    preprocessed_image = load_and_preprocess_data(image_path)

    predictions = model.predict(preprocessed_image)
    score = predictions[0].tolist()  # Convert the NumPy array to a Python list

    # Finding the index of the class with the highest probability
    predicted_class_index = score.index(max(score))

    class_labels = [
        "American football", "Baseball", "Basketball", "Billiard", "Bowling",
        "Cricket", "Football", "Golf", "Hockey ball", "Hockey puck", "Rugby",
        "Shuttlecock", "Table tennis", "Tennis", "Volleyball",
    ]
    predicted_class = class_labels[predicted_class_index]

    # Construct a JSON object with the predicted class and its corresponding probability
    result = {
        "Predicted Class": predicted_class,
        "Probability": f"{round(max(score), 4) * 100} %"
    }
    # Convert the result to a JSON string
    result_json = json.dumps(result)
    return predicted_class, result["Probability"]
