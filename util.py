import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

path = 'C://Users//karth//OneDrive//Desktop//GROUND NUT LEAVES DETECTION//Raw_Data//rust'

print(len(os.listdir(path)))

inp_dir = 'C://Users//karth//OneDrive//Desktop//GROUND NUT LEAVES DETECTION//Raw_Data'

target_size = (256, 256)

print(os.listdir(inp_dir))

for folder in os.listdir(inp_dir):


  folder_path = os.path.join(inp_dir, folder)
  if os.path.isdir(folder_path):

    print(f'Processing images in the folder: {folder}')
    for filename in os.listdir(folder_path):

      if filename.endswith(('.jpg', '.jpeg', '.png', ".JPG")):



        #load the img
        img_path = os.path.join(folder_path, filename)
        img = mpimg.imread(img_path)

        #resizing
        resized_img = cv2.resize(img, target_size)

        #overwriting the input images with the resized images
        mpimg.imsave(img_path, resized_img)


print('Done with Resizing')

img_path = 'C://Users//karth//OneDrive//Desktop//GROUND NUT LEAVES DETECTION//Raw_Data//early_leaf_spot//IMG_0325.JPG'
img = mpimg.imread(img_path)
print(img.shape)
plt.imshow(img)
plt.axis('off')
plt.show()