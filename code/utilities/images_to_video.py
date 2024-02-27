"""
Script Title: Images to Video
Author: Petter Hangerhagen
Email: petthang@stud.ntnu.no
Date: February 27, 2024
Description: This script contains support functions for converting images to video in the video.py file.
"""

import cv2
import os
import re

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


def images_to_video_opencv(path, output_name='video.avi', fps=1):
    images = [img for img in os.listdir(path) if img.endswith(".png")]
    sort_nicely(images)

    frame = cv2.imread(os.path.join(path, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(path, image)))

    cv2.destroyAllWindows()
    video.release()



def empty_folder(dir_path):
    for f in os.listdir(dir_path):
        os.remove(os.path.join(dir_path, f))


