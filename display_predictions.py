from bs4 import BeautifulSoup
import time
import pandas as pd
import streamlit as st
import pytesseract
from PIL import ImageGrab
import pickle

rf = pickle.load(open("xBases_model.pkl", "rb"))

pytesseract.pytesseract.tesseract_cmd = 'usr/bin/tesseract'

def capture_screenshot():
    exit_velo_screenshot = ImageGrab.grab(bbox=(860, 260, 930, 310))
    exit_velo = pytesseract.image_to_string(exit_velo_screenshot)

    angle_screenshot = ImageGrab.grab(bbox=(1020, 260, 1090, 310))
    angle = pytesseract.image_to_string(angle_screenshot)

    direction_screenshot = ImageGrab.grab(bbox=(1180, 260, 1250, 310))
    direction = pytesseract.image_to_string(direction_screenshot)

    distance_screenshot = ImageGrab.grab(bbox=(1340, 260, 1410, 310))
    distance = pytesseract.image_to_string(distance_screenshot)

    ExitSpeed = int(exit_velo)
    Angle = int(angle)
    Direction = int(direction)
    Distance = int(distance)

    return (ExitSpeed, Angle, Direction, Distance)
    
def predict_bases(ExitSpeed, Angle, Direction, Distance):
    hit_metrics = pd.DataFrame([[ExitSpeed, Angle, Direction, Distance]], columns=['ExitSpeed', 'Angle', 'Direction', 'Distance'])
    prediction = rf.predict(hit_metrics)
    prediction = prediction[0]
    
    if prediction == 0:
        prediction = "Out"
    elif prediction == 1:
        prediction = "Single"
    elif prediction == 2:
        prediction = "Double"
    elif prediction == 3:
        prediction = "Triple"
    elif prediction == 4:
        prediction = "Home Run"
    
    return prediction

def main():
    time.sleep(1)
    capture_screenshot()

if __name__ == "__main__":
    main()