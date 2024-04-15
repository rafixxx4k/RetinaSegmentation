import tkinter as tk
import os
import numpy as np
from tkinter import filedialog
from PIL import Image, ImageTk
from accuracy import accuracy


class ImageLoaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Loader App")
        self.for3 = None
        self.for4 = None
        self.for5 = None
        path = os.path.join("DRIVE", "training")
        self.retina_image = None
        self.mask = None
        self.manual = None
        self.training_images = []
        self.training_mask = []
        self.training_manual = []
        self.test_images = []
        self.test_mask = []
        for file in os.listdir(os.path.join(path, "images")):
            self.training_images.append(os.path.join(path, "images", file))
        for file in os.listdir(os.path.join(path, "mask")):
            self.training_mask.append(os.path.join(path, "mask", file))
        for file in os.listdir(os.path.join(path, "1st_manual")):
            self.training_manual.append(os.path.join(path, "1st_manual", file))
        path = os.path.join("DRIVE", "test")
        for file in os.listdir(os.path.join(path, "images")):
            self.test_images.append(os.path.join(path, "images", file))
        for file in os.listdir(os.path.join(path, "mask")):
            self.test_mask.append(os.path.join(path, "mask", file))
        self.training_images = sorted(self.training_images)
        self.training_mask = sorted(self.training_mask)
        self.training_manual = sorted(self.training_manual)
        self.test_images = sorted(self.test_images)
        self.test_mask = sorted(self.test_mask)
        # Variables
        self.image_path = os.path.join("DRIVE", "training", "images", "21_training.tif")
        self.image_label = tk.Label(root, text="Image from 1-20:")
        self.image_entry = tk.Entry(
            root,
        )
        self.load_button = tk.Button(root, text="Load Images", command=self.load_images)
        self.image_frame = tk.Frame(root)

        self.mode_label = tk.Label(root, text="Mode:")
        self.mode_var = tk.StringVar()
        self.train_radio = tk.Radiobutton(
            root, text="Training Set", variable=self.mode_var, value="train"
        )
        self.test_radio = tk.Radiobutton(
            root, text="Test Set", variable=self.mode_var, value="test"
        )
        self.train_radio.select()
        self.bottom_frame = tk.Frame(root)
        self.bottom_frame.grid(row=4, column=0, columnspan=3)
        self.image_labels = [[None for _ in range(3)] for _ in range(2)]
        self.create_images()
        self.load_buttons()

        # Packing
        self.image_label.grid(row=0, column=0)
        self.image_entry.grid(row=0, column=1)
        self.load_button.grid(row=0, column=2)
        self.mode_label.grid(row=1, column=0)
        self.train_radio.grid(row=1, column=1)
        self.test_radio.grid(row=1, column=2)
        self.image_frame.grid(row=2, column=0, columnspan=3)
        self.bottom_frame.grid(row=3, column=0, columnspan=3)

    def create_images(self):
        # Function to load images into the grid
        for widget in self.image_frame.winfo_children():
            widget.destroy()

        captions = [
            ["Orginal image", "Image mask", "Manual marked vessel"],
            ["na 3", "na 4", "na 5"],
        ]  # Example captions

        for i in range(2):
            for j in range(3):
                # image = Image.open(self.image_path)
                # image = image.resize((150, 150))
                photo = ImageTk.PhotoImage(
                    Image.fromarray(np.zeros((200, 200), dtype=np.uint8))
                )
                caption_label = tk.Label(self.image_frame, text=captions[i][j])
                caption_label.grid(row=i * 2, column=j)
                label = tk.Label(self.image_frame, image=photo)
                label.image = photo
                label.grid(row=i * 2 + 1, column=j)
                self.image_labels[i][j] = label

    def load_buttons(self):
        # Function to load buttons
        self.first_button = tk.Button(
            self.bottom_frame, text="Image Processing", command=self.process_image
        )
        self.first_button.grid(row=0, column=0, pady=(10, 10))
        self.second_button = tk.Button(
            self.bottom_frame, text="K-Nearest Neighbors", command=self.knn
        )
        self.second_button.grid(row=0, column=1, pady=(10, 10))
        self.third_button = tk.Button(
            self.bottom_frame, text="Test Model", command=self.test_model
        )
        self.third_button.grid(row=0, column=2, pady=(10, 10))

    def update_image(self, i, j, new_image):
        # Function to update a specific image in the grid
        photo = ImageTk.PhotoImage(Image.fromarray(new_image).resize((200, 200)))
        label = self.image_labels[i][j]
        label.configure(image=photo)
        label.image = photo

    def load_images(self):
        try:
            image_number = int(self.image_entry.get()) - 1
        except ValueError:
            self.image_entry.delete(0, "end")
            return
        if image_number < 0 or image_number > 19:
            self.image_entry.delete(0, "end")
            return
        if self.mode_var.get() == "train":
            self.retina_image = np.array(Image.open(self.training_images[image_number]))
            self.mask = np.array(Image.open(self.training_mask[image_number]))
            self.manual = np.array(Image.open(self.training_manual[image_number]))
            self.update_image(0, 0, self.retina_image)
            self.update_image(0, 1, self.mask)
            self.update_image(0, 2, self.manual)
        else:
            self.retina_image = np.array(Image.open(self.test_images[image_number]))
            self.mask = np.array(Image.open(self.test_mask[image_number]))
            self.manual = None
            self.update_image(0, 0, self.retina_image)
            self.update_image(0, 1, self.mask)
            self.update_image(0, 2, np.zeros((200, 200), dtype=np.uint8))
        root.update()

    def process_image(self):
        from na3 import find_vessels

        if self.retina_image is None:
            return
        self.for3 = find_vessels(self.retina_image, self.mask)
        self.update_image(1, 0, self.for3)

        if self.manual is not None:
            print(f"na 3 image{self.image_entry.get()}:", end="\t")
            accuracy(self.for3.flatten(), self.manual.flatten())
        else:
            print("Manual not available")

    def knn(self):
        from na4 import find_vessels

        if self.retina_image is None:
            return
        self.for4 = find_vessels(self.retina_image, self.mask)
        self.update_image(1, 1, self.for4)

        if self.manual is not None:
            print(f"na 4 image{self.image_entry.get()}:", end="\t")
            accuracy(self.for4.flatten(), self.manual.flatten())
        else:
            print("Manual not available")

    def test_model(self):
        from na5 import find_vessels

        if self.retina_image is None:
            return
        self.for5 = find_vessels(self.retina_image, self.mask)
        self.update_image(1, 2, self.for5)

        if self.manual is not None:
            print(f"na 5 image{self.image_entry.get()}:", end="\t")
            accuracy(self.for5.flatten(), self.manual.flatten())
        else:
            print("Manual not available")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLoaderApp(root)
    root.mainloop()
