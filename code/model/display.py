import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image

def display_images_from_folder(folder_path, num_images):
    images = os.listdir(folder_path)
    num_images = min(num_images, len(images))

    for i in range(num_images):
        image_path = os.path.join(folder_path, images[i])
        image = cv2.imread(image_path)
        
        # Display the image
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()