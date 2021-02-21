import cv2
from tqdm import tqdm
import numpy as np

FILE_PATH = "C:/Users/mark/OneDrive/Documents/Blender Projects/outputs/"
KP_LIST = ["kp_0_", "kp_1_", "kp_2_", "kp_3_"]
N_IMAGES = 225

def extract_circle(img):
    """extracts the centre point and radius of the first detected circle

    Args:
        img ([cv2]): grayscale input image

    Returns:
        circle [list len=3]: y,x image coords of circle centre, radius in pixels
    """
    ret, thresh = cv2.threshold(img, 177, 200, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    circle = cv2.minEnclosingCircle(cnt)
    return circle

def display_result(img,circle):
    """display circle image with result

    Args:
        img ([cv2 image]): image to draw on
        circle ([from extract circles]): circle to draw
    """
    color_im = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    center_tup = (tuple(np.asarray(circle[0]).astype(np.int16)))
    # edge
    cv2.circle(color_im,center_tup,int(circle[1]),(0,255,0),2)
    # centre
    cv2.circle(color_im,center_tup,2,(0,0,255),3)
    cv2.imshow('detected circles',color_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_single_image():
    """load a single keypoint image and display extracted circle with centrepoint
    """
    img = cv2.imread('C:/Users/mark/OneDrive/Documents/Blender Projects/outputs/kp_0_0001.png',0)
    cv2.imshow('loaded image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    circ = extract_circle(img)
    display_result(img,circ)
  
def main():
    print("loading keypoint images from {}".format(FILE_PATH))
    for i in tqdm(range(N_IMAGES)):
        kps = []
        for kp_name in KP_LIST:
            im = cv2.imread( FILE_PATH + kp_name + str(i+1).zfill(4) + '.png',0)
            circ = extract_circle(im)
            kps.append(circ[0])
        np.save( FILE_PATH + "kps_" + str(i+1).zfill(4) + ".npy", np.asarray(kps))
        

if __name__ == "__main__":
    main()
