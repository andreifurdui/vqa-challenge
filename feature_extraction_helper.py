import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras import backend as K

K.set_image_data_format('channels_first')


def image_feature_extractor(image_frame, model):
    """
    Extracts (512, 7, 7)-dimensional CNN features

    Input:
        image_file: image filenames

    Returns:
        (512, 7, 7)-dimensional CNN features
    """
    # crop central part of the image and resize, preserving as much as possible without padding or changing aspect ratio
    img = tf.image.crop_to_bounding_box(image_frame, 0, 184, 480, 480)
    img = tf.image.resize(img, [224, 224])
    img = image.img_to_array(img)

    # preprocess the image by
    # (1) expanding the dimensions to include batch dim and
    # (2) subtracting the mean RGB pixel intensity from the ImageNet dataset
    img = tf.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # pass the images through the network and use the outputs as our actual features
    features = model.predict(img)  # (BATCH_SIZE, 512, 7, 7)
    features = tf.reshape(features, (features.shape[0], features.shape[1], -1))  # (BATCH_SIZE, 512, 49)
    features = tf.transpose(features, perm=[0, 2, 1])  # (BATCH_SIZE, 49, 512)
    return features
