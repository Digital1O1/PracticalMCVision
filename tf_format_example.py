import tensorflow as tf
import os
from PIL import Image
import numpy as np


# Function to convert image to bytes
def _bytes_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )


# Function to create an example for a single image
def create_example(image_path, label):
    image = np.array(Image.open(image_path))
    feature = {
        "image": _bytes_feature(image),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


# Directory containing your images
image_dir = "/path/to/your/images/"

# TFRecord file to be created
output_file = "images.tfrecord"

# List of image files and corresponding labels
image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
labels = [0, 1, 0, 1, 0]  # Replace with your actual labels

# Open a TFRecord writer
with tf.io.TFRecordWriter(output_file) as writer:
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)
        label = labels[i]

        # Create a TFRecord example
        example = create_example(image_path, label)

        # Serialize the example and write it to the TFRecord file
        writer.write(example.SerializeToString())

print(f"TFRecord file {output_file} created successfully.")
