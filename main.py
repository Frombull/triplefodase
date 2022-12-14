import datetime
import cv2
import keyboard
import numpy as np
import pyautogui
from PIL import Image
from mss import mss
from enum import Enum
from constants import *

Screen = Enum('Screen', 'match_registration deck_selection playing end')


def log(msg: str):
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    print(f'[{current_time}] - {msg}')


def screenshot(resolution: dict) -> Image:
    sct_img = mss().grab(resolution)
    img = Image.frombytes('RGB', (sct_img.size.width, sct_img.size.height), sct_img.rgb)
    return img


def get_mask(frame_hsv, color: str) -> Image:
    if (color == 'blue'):
        color_min = BLUE_MIN
        color_max = BLUE_MAX
    elif (color == 'green'):
        color_min = GREEN_MIN
        color_max = GREEN_MAX
    else:
        color_min = YELLOW_MIN
        color_max = YELLOW_MAX

    low_range = np.array(color_min)
    high_range = np.array(color_max)
    green_mask = cv2.inRange(frame_hsv, low_range, high_range)

    return green_mask


def on_challange_screen(frame_bgr) -> bool:
    button = cv2.imread(CHALLANGE_BUTTON_SAMPLE)

    button_match = cv2.matchTemplate(frame_bgr, button, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(button_match)

    log(f'Max val: {max_val} ||| Max loc: {max_loc}')

    return max_val >= 0.9


def get_card_xy(frame_mask):
    contours, hierarchy = cv2.findContours(frame_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    M = cv2.moments(contours[0])

    if M["m00"] == 0:
        return None

    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return [cX, cY]


def play_card(blue_mask, green_mask, yellow_mask):
    blue_pos = get_card_xy(blue_mask)
    yellow_pos = get_card_xy(yellow_mask)
    green_pos = get_card_xy(green_mask)

    if blue_pos is None or green_pos is None:
        return

    pyautogui.moveTo(green_pos[0], green_pos[1])
    pyautogui.click()
    pyautogui.moveTo(blue_pos[0], blue_pos[1])
    pyautogui.click()


# def adjust_size(frame_bgr) -> Image:
#    start_point = (,)
#    end_point = (,)
#    color = (255, 0, 0)  # BGR
#    thickness = 2
#
#    frame_bgr = cv2.rectangle(frame_bgr, start_point, end_point, color, thickness)
#
#    cv2.imshow("Original (BGR)", frame_bgr)
#    cv2.waitKey(1)
#
#    return frame_bgr

def registration_window(frame_bgr) -> Image:
    frame_original = frame_bgr
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    window_template = cv2.imread('images/match_registration.jpg', 0)
    button_template = cv2.imread('images/challenge_button.jpg', 0)

    result = cv2.matchTemplate(frame_gray, window_template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    threshold = 0.92
    if(max_val < threshold):
        return frame_bgr

    ## Find registration window pos
    top_left = max_loc
    height, width = window_template.shape[:2]
    bottom_right = (top_left[0] + width, top_left[1] + height)

    ## Write on image
    text_bottom_left = (top_left[0], top_left[1] - 12)
    cv2.putText(frame_bgr, 'Registration Window', text_bottom_left, cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(250, 0, 10), thickness=1, lineType=2)
    cv2.rectangle(frame_bgr, top_left, bottom_right, (250, 0, 10), 5)

    ## Cropped image
    print(f'{top_left[0]} | {top_left[1]} ||||| {bottom_right[0]} | {bottom_right[1]}')
    cropped = frame_original[top_left[0]:top_left[1], bottom_right[0]:bottom_right[1]]

    ## Find button pos
    # result = cv2.matchTemplate(frame_gray, button_template, cv2.TM_CCOEFF_NORMED)
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    #
    # threshold = 0.92
    # if(max_val < threshold):
    #     return frame_bgr



    return cropped


def main():
    while True:
        ##Take screenshot
        frame = screenshot(resolution=SCREEN_RES)

        ##Convert to BGR and HSV
        frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        frame_hsv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2HSV)

        ## Get masks
        blue_mask = get_mask(frame_hsv, 'blue')
        green_mask = get_mask(frame_hsv, 'green')
        yellow_mask = get_mask(frame_hsv, 'yellow')

        ## Draw center of cards
        # cv2.circle(frame_bgr, (cX, cY), 20, (0, 0, 250), 20)

        ##Resize images
        #frame_bgr = cv2.resize(frame_bgr, None, fx=0.5, fy=0.5)
        #frame_hsv = cv2.resize(frame_hsv, None, fx=0.5, fy=0.5)

        ##
        registration_window_image = registration_window(frame_bgr)

        ##Show images
        #cv2.imshow("Blue mask", blue_mask)
        #cv2.imshow("Green mask", green_mask)
        #cv2.imshow("Original (BGR)", frame_bgr)
        cv2.imshow("Registration Window", registration_window_image)
        cv2.waitKey(1)

        ##Stop the bot
        if keyboard.is_pressed('q'):
            log('Stopping bot.')
            cv2.destroyAllWindows()
            quit()
        if keyboard.is_pressed('r'):
            play_card(blue_mask, green_mask, yellow_mask)
        if keyboard.is_pressed('t'):
            pass


if __name__ == '__main__':
    main()
