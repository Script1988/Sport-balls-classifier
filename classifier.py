import tensorflow as tf


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
    score = predictions[0]

    return score
