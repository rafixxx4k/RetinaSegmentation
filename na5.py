import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import matplotlib.pyplot as plt
import joblib
import os


class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        self.augment_inputs = tf.keras.layers.RandomFlip("horizontal", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip("horizontal", seed=seed)

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels


def find_vessels(retina_image, mask):

    if not os.path.exists("model.keras"):
        print("Model not found, please train the model first.")
        return

    model = tf.keras.models.load_model("model.keras")
    retina_image, mask = load_image(retina_image, mask)
    retina_image = tf.expand_dims(retina_image, axis=0)
    prediction = model.predict(retina_image)
    prediction = tf.math.argmax(prediction, axis=-1)
    prediction = tf.squeeze(prediction)

    prediction = tf.image.resize(
        prediction[..., tf.newaxis],
        (584, 565),
        method="bilinear",
    )
    prediction = tf.squeeze(prediction)
    prediction = tf.cast(prediction, bool)
    prediction = prediction.numpy().astype(bool)
    return prediction


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


def load_image(image_path, mask_path):

    retina_image = cv2.imread(image_path)
    retina_image = cv2.cvtColor(retina_image, cv2.COLOR_BGR2RGB)
    mask = np.array(Image.open(mask_path))

    retina_image, mask = normalize(retina_image, mask)
    retina_image, mask = resize(retina_image, mask, size=(512, 512))

    return retina_image, mask


def load_training_set():

    # Load the training data
    retina_images = []
    masks = []

    for i in range(21, 41):
        image_path = f"DRIVE/training/images/{i}_training.tif"
        mask_path = f"DRIVE/training/1st_manual/{i}_manual1.gif"

        retina_image, mask = load_image(image_path, mask_path)

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


def display(retina_image, true_mask, pred_mask=None):

    figs = 3 if pred_mask is not None else 2
    plt.figure(figsize=(figs * 6, 6))

    plt.subplot(1, figs, 1)
    plt.title("Retina Image")
    plt.imshow(tf.squeeze(retina_image).numpy(), cmap="gray")

    plt.subplot(1, figs, 2)
    plt.title("True Mask")
    plt.imshow(tf.squeeze(true_mask).numpy(), cmap="gray")

    if pred_mask is not None:
        plt.subplot(1, figs, 3)
        plt.title("Predicted Mask")
        plt.imshow(tf.squeeze(pred_mask).numpy(), cmap="gray")

    plt.savefig("sample.png")
    plt.show()


def prepare_training_data():

    # Load the images
    retina_images, masks = load_training_set()

    # Split the data into training and testing sets
    train_indices = np.random.rand(len(retina_images)) < 0.7
    x_train = np.array(retina_images)[train_indices]
    y_train = np.array(masks)[train_indices]
    x_test = np.array(retina_images)[~train_indices]
    y_test = np.array(masks)[~train_indices]

    # Prepare training and testing batches
    train_batches = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .cache()
        .shuffle(len(x_train))
        .batch(3)
        .repeat()
        .map(Augment())
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    test_batches = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(2).repeat()
    )

    return train_batches, test_batches


def train_model():

    train_batches, test_batches = prepare_training_data()

    # Create and train the model
    model = create_model()
    model.fit(
        train_batches,
        epochs=10,
        steps_per_epoch=5,
        validation_data=test_batches,
        validation_steps=5,
    )
    model.save("model.keras")

    # Evaluate the model
    for image, mask in test_batches.take(1):
        predictions = model.predict(image).squeeze()
        predictions = tf.math.argmax(predictions, axis=-1)
        predictions = predictions[..., tf.newaxis]
        display(image[0], mask[0], predictions[0])
        break


def create_model():

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(512, 512, 3), include_top=False
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

    # Save the model architecture to a file
    tf.keras.utils.plot_model(
        model, show_shapes=True, expand_nested=False, dpi=128, to_file="model.png"
    )

    return model


def unet_model(output_channels, down_stack, up_stack):

    inputs = tf.keras.layers.Input(shape=[512, 512, 3])
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
    # np.random.seed(42)
    train_model()
