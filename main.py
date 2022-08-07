import datetime
import time

import cv2
import keyboard
import mouse
import numpy as np
import pydirectinput
import pytesseract
from PIL import Image
from mss import mss

from constants import *


def log(msg: str):
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    print(f'[{current_time}] - {msg}')


def screenshot(resolution: dict) -> Image:
    sct_img = mss().grab(resolution)
    img_raw = Image.frombytes('RGB', (sct_img.size.width, sct_img.size.height), sct_img.rgb)
    return img_raw


def test_blue():
    log('testing blue')

    img = cv2.imread('images/card_blue.jpg')
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ## Mask of blue
    blue_mask = cv2.inRange(img_hsv, BLUE_MIN, BLUE_MAX)

    ## Final mask and masked
    target = cv2.bitwise_and(img, img, mask=blue_mask)

    cv2.imwrite("result.png", target)
    log('done testing')


def main():
    test_blue()
    '''
    while True:
        ##Take screenshot
        img_raw = screenshot(resolution=SCREEN_RES)

        ##Convert to BGR and HSV
        img_bgr = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
        img_hsv = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2HSV)

        ##Resize image to 80%
        img_bgr = cv2.resize(img_bgr, None, fx=0.8, fy=0.8)
        img_hsv = cv2.resize(img_hsv, None, fx=0.8, fy=0.8)

        ##Show images
        cv2.imshow("Original Image (BGR)", img_bgr)
        cv2.waitKey(2)

        ##Stop the bot
        if keyboard.is_pressed('q'):
            log('Stopping bot.')
            cv2.destroyAllWindows()
            quit()
    '''


if __name__ == '__main__':
    log('Starting bot')
    main()
