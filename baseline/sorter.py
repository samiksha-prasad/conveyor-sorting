import cv2
import numpy as np
import serial
import time

# Initialize Arduino Serial Communication
arduino = serial.Serial(port='COM4', baudrate=9600, timeout=.1)  # Replace COM3 with your Arduino port

def detect_color(image):
    """
    Detect the predominant color in the image.
    Returns the color as a string: 'Red', 'Green', 'Blue', or 'Unknown'.
    """
    # Resize and blur the image
    resized = cv2.resize(image, (100, 100))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)

    # Convert to HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for colors
    color_ranges = {
        'Red': [(0, 120, 70), (10, 255, 255)],
        'Green': [(36, 50, 70), (89, 255, 255)],
        'Blue': [(90, 50, 70), (128, 255, 255)]
    }

    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        if cv2.countNonZero(mask) > 1000:  # Threshold to detect significant color
            return color

    return 'Unknown'

def send_command_to_arduino(color):
    """
    Send a command to Arduino based on detected color.
    """
    color_to_command = {
        'Red': 'R',  # Command for Red
        'Green': 'G',  # Command for Green
        'Blue': 'B'   # Command for Blue
    }
    command = color_to_command.get(color, None)
    if command:
        arduino.write(command.encode())
        print(f"Command sent to Arduino: {command}")
        time.sleep(1)

# Initialize camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Frame", frame)

    # Detect color
    detected_color = detect_color(frame)
    print(f"Detected Color: {detected_color}")

    # Send command to Arduino
    send_command_to_arduino(detected_color)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()