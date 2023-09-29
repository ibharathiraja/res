import os
import time
from PIL import Image
import numpy as np
import tensorflow as tf

def preprocess_image(image_path):
    hr_image = tf.image.decode_image(tf.io.read_file(image_path))
    # If PNG, remove the alpha channel. The model only supports
    # images with 3 color channels.
    if hr_image.shape[-1] == 4:
        hr_image = hr_image[...,:-1]
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)

# Defining helper functions
def downscale_image(image):
    image_size = []
    if len(image.shape) == 3:
        image_size = [image.shape[1], image.shape[0]]
    else:
        raise ValueError("Dimension mismatch. Can work only on single image.")
    image = tf.squeeze(
            tf.cast(
                    tf.clip_by_value(image, 0, 255), tf.uint8))
    lr_image = np.asarray(
        Image.fromarray(image.numpy())
        .resize([image_size[0] // 4, image_size[1] // 4],
                            Image.BICUBIC))
    lr_image = tf.expand_dims(lr_image, 0)
    lr_image = tf.cast(lr_image, tf.float32)
    return lr_image

def save(down_scaled_image,name):
    image = np.asarray(tf.squeeze(down_scaled_image))
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save(name)

def inference(pre_processed_img, output_img_name):
    start = time.time()
    predicted_img = model(pre_processed_img)
    predicted_img_ = tf.squeeze(predicted_img)
    print("Time Taken: %f" % (time.time() - start))
    image = np.asarray(predicted_img_)
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save(output_img_name)

!wget "https://lh4.googleusercontent.com/-Anmw5df4gj0/AAAAAAAAAAI/AAAAAAAAAAc/6HxU8XFLnQE/photo.jpg64" -O test.jpg
model = tf.keras.models.load_model("model")

input_img_path="/content/test.jpg"
downscaled_img_path=input_img_path[:-4]+"_downscaled.jpg"
output_img_name = input_img_path[:-4]+"_output.jpg"

img_pre_process = preprocess_image(input_img_path)
down_scaled_image = downscale_image(tf.squeeze(img_pre_process))
save(down_scaled_image, downscaled_img_path)
inference(down_scaled_image,output_img_name)
