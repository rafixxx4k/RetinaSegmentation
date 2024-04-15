import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import matplotlib.pyplot as plt


def find_vessels(retina_image, mask):
    print("Finding vessels...")


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = tf.cast(input_mask, tf.int32) / 255
    return input_image, input_mask


def resize(input_image, input_mask, size=(128, 128)):
    input_image = tf.image.resize(input_image, size)
    input_mask = tf.image.resize(
        input_mask[..., tf.newaxis], size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    input_mask = tf.squeeze(input_mask)

    return input_image, input_mask


def load_images():

    # Load the training data
    retina_images = []
    masks = []

    for i in range(21, 41):
        image_path = f"DRIVE/training/images/{i}_training.tif"
        mask_path = f"DRIVE/training/1st_manual/{i}_manual1.gif"

        retina_image = cv2.imread(image_path)
        retina_image = cv2.cvtColor(retina_image, cv2.COLOR_BGR2RGB)
        mask = np.array(Image.open(mask_path))

        retina_image, mask = normalize(retina_image, mask)
        retina_image, mask = resize(retina_image, mask)

        if retina_image is None:
            print(f"Failed to load image: {image_path}")
        else:
            retina_images.append(retina_image)

        if mask is None:
            print(f"Failed to load mask: {mask_path}")
        else:
            masks.append(mask)

    print(len(retina_images), "retina images loaded.")
    print(len(masks), "masks loaded.")

    return retina_images, masks


def display_sample(retina_image, true_mask, pred_mask=None):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Retina Image")
    plt.imshow(retina_image.numpy().squeeze(), cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("True Mask")
    plt.imshow(true_mask.numpy().squeeze(), cmap="gray")
    plt.savefig("sample.png")
    # plt.show()


def display_prediction(retina_image, true_mask, pred_mask):
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.title("Retina Image")
    plt.imshow(tf.squeeze(retina_image).numpy(), cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("True Mask")
    plt.imshow(tf.squeeze(true_mask).numpy(), cmap="gray")

    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(tf.squeeze(pred_mask).numpy(), cmap="gray")

    plt.savefig("sample_comparison.png")
    plt.show()


# def create_mask(pred_mask):
#     pred_mask = tf.math.argmax(pred_mask, axis=-1)
#     pred_mask = pred_mask[..., tf.newaxis]
#     return pred_mask[0]


# def show_predictions(dataset=None, num=1):
#     if dataset:
#         for image, mask in dataset.take(num):
#             pred_mask = model.predict(image)
#             display_sample([image[0], mask[0], create_mask(pred_mask)])
#     else:
#         display_sample(
#             [
#                 sample_image,
#                 sample_mask,
#                 create_mask(model.predict(sample_image[tf.newaxis, ...])),
#             ]
#         )


def train_model():

    retina_images, masks = load_images()
    random_index = np.random.randint(0, len(retina_images))
    display_sample(retina_images[random_index], masks[random_index])

    train_indices = np.random.rand(len(retina_images)) < 0.7

    x_train = np.array(retina_images)[train_indices]
    y_train = np.array(masks)[train_indices]
    x_test = np.array(retina_images)[~train_indices]
    y_test = np.array(masks)[~train_indices]

    print("Training data:", x_train.shape, y_train.shape)
    print("Test data:", x_test.shape, y_test.shape)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=x_train[0].shape, include_top=False
    )

    layer_names = [
        "block_1_expand_relu",  # 64x64
        "block_3_expand_relu",  # 32x32
        "block_6_expand_relu",  # 16x16
        "block_13_expand_relu",  # 8x8
        "block_16_project",  # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),  # 32x32 -> 64x64
    ]

    model = unet_model(2, down_stack, up_stack)
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    tf.keras.utils.plot_model(
        model, show_shapes=True, expand_nested=False, dpi=128, to_file="model.png"
    )

    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    predictions = model.predict(x_test)

    # print("Predictions shape:", predictions.shape)
    print("True mask shape:", y_test.shape)

    print("Predictions shape:", predictions.shape)
    # print("Predictions:", predictions[0])
    # print("True mask:", y_test[0])
    predictions = tf.math.argmax(predictions, axis=-1)
    predictions = predictions[..., tf.newaxis]
    print("Predictions shape:", predictions.shape)
    # print("Predictions:", predictions[0])
    predictions = tf.squeeze(predictions)
    print("Predictions shape:", predictions.shape)

    # print("true mask: ", y_test[0])
    # print("predictions: ", predictions[0])
    for i in range(128):
        for j in range(128):
            print(y_test[0][i][j], predictions[0][i][j])

    for i in range(2):
        display_prediction(x_test[i], y_test[i], predictions[i])


def unet_model(output_channels, down_stack, up_stack):

    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2, padding="same"
    )

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


if __name__ == "__main__":
    # Set seed for reproducibility, the number was chosen by @rafixxx4k, DM him for more information
    np.random.seed(42)
    train_model()
