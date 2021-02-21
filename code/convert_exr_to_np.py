import cv2
import numpy as np
from tqdm import tqdm

FILE_PATH = "C:/Users/mark/OneDrive/Documents/Blender Projects/outputs/"
N_IMAGES = 225

for i in tqdm(range(N_IMAGES)):
    # open exr image and grab first channel (all 3 channels are the same)
    a = cv2.imread(FILE_PATH+"depth"+str(i+1).zfill(4)+".exr",  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,0]
    # convert to 1/10th mm values and save as int16
    a*=10000
    np.save(FILE_PATH+"depth_"+str(i+1).zfill(4)+".npy", a.astype(np.int16))

