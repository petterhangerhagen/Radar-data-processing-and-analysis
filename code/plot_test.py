# import cv2
# import os

# # Path to the directory containing your PNG files
# image_folder = 'path/to/your/png/files'

# # Sort the files in case they're not in the correct order
# images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])

# # Video settings
# frame = cv2.imread(os.path.join(image_folder, images[0]))
# height, width, layers = frame.shape
# video_name = 'output_video.mp4'
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID' for .avi format

# # Initialize the video writer
# out = cv2.VideoWriter(video_name, fourcc, 30, (width, height))

# # Loop through the images and add them to the video
# for image in images:
#     img_path = os.path.join(image_folder, image)
#     frame = cv2.imread(img_path)
#     out.write(frame)

# # Release the video writer
# out.release()
from tqdm import tqdm
from time import sleep
# my_list = [1,2,3,4,5,6,7,8,9]
# #### In case using with enumerate:
# for i, x in enumerate( tqdm(my_list) ):
#     sleep(1)
#     # do something with i and x

import progressbar

members = [1,2,3,4,5,6,7,8,9]

bar = progressbar.ProgressBar(maxval=len(members)).start()

for idx, member in enumerate(members):
    sleep(1)
    bar.update(idx)