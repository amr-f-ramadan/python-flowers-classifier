# predict.py
import argparse, json
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

image_size = 224

def process_image(image):
    image = tf.image.decode_image(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image.numpy()

def predict(image_path, model_path, top_k=5, category_names=None):
    """Predict the class of the flower in the image."""

    # Load model
    URL = "https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/feature_vector/5"
    feature_extractor = hub.KerasLayer(URL, input_shape=(image_size, image_size, 3))

    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': feature_extractor})
   
    # Preprocess image
    im = tf.io.read_file(image_path)
    test_im = np.asarray(im)
    processed_im = process_image(test_im)
    input_image = np.expand_dims(processed_im, axis=0)

    # Get predictions
    preds = model.predict(input_image)[0]
    sorted_indices = np.argsort(preds)
    top_classes = sorted_indices[-top_k:]
    top_probs = preds[top_classes]

    # Map class indices to names if category_names is provided
    if category_names:
        with open(category_names, 'r') as f:
            label_map = json.load(f)
        top_class_names = [label_map[str(cls)] for cls in top_classes]
        return top_probs, top_class_names
    else:
        return top_probs, top_classes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict the flower type from an image.')
    parser.add_argument('image_path', type=str, help='Path to the image')
    parser.add_argument('model_path', type=str, help='Path to the saved Keras model')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping labels to flower names')

    args = parser.parse_args()

    top_probs, top_classes = predict(args.image_path, args.model_path, args.top_k, args.category_names)

    # Output predictions
    print(f"Top {args.top_k} predictions:")
    for i in range(len(top_probs) - 1, -1, -1):
        print(f"Class: {top_classes[i]} - Probability = {top_probs[i]:.3%}")