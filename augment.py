# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 00:04:15 2024

@author: karth
"""

import os
import imgaug.augmenters as iaa
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# img = cv2.imread('C://Users//karth//OneDrive//Desktop//GROUND NUT LEAVES DETECTION//Raw_Data//early_leaf_spot//IMG_0326.JPG')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # cv2.imsow("Image", img)
# plt.imshow(img)

# Loading the dataset
images_path = glob.glob('C://Users//karth//OneDrive//Desktop//GROUND NUT LEAVES DETECTION//Raw_Data//rust//*.JPG')
# print(images_path)

images = []

for img_path in images_path:
    image = mpimg.imread(img_path)
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()
    # plt.imshow(img_path)
    images.append(image)
    
# print(images)
# plt.imshow(images[0])
# Image Augmentation

augmentation = iaa.Sequential([
        # iaa.Fliplr(1)
        # iaa.Flipud(1)
        # iaa.Affine(shear=(-20, 20))
        iaa.Affine(scale=(0.8, 1.2))
    ])

augmented_images = augmentation(images=images)


# Show the imagges
# plt.imshow(augmented_images[69])
# plt.axis('off')

# Save the augmented images
# Output directory to save augmented images
output_dir = 'C://Users//karth//OneDrive//Desktop//GROUND NUT LEAVES DETECTION//Augmented_Data//rust//'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate over augmented images and save them
for i, augmented_image in enumerate(augmented_images):
    output_path = os.path.join(output_dir, f'augmented_image_zoom_{i}.jpg')  # Generate unique filename
    mpimg.imsave(output_path, augmented_image)  # Save augmented image
    print(f'Saved augmented image {i+1} at {output_path}')

print('All augmented images saved successfully.')