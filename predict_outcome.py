import pickle
import pandas as pd

rf = pickle.load(open("xBases_model.pkl", "rb"))

print("\n")

while True:

    exit_velo = input("Enter exit velocity: ")
    launch_angle = input("Enter launch angle: ")
    distance = input("Enter distance: ")
    direction = input("Enter direction: ")

    hit_metrics = pd.DataFrame([[exit_velo, launch_angle, direction, distance]], columns=['ExitSpeed', 'Angle', 'Direction', 'Distance'])

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

    print(f"\nResult: {prediction}\n")