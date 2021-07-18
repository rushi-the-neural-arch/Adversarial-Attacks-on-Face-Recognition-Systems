import cv2
import mediapipe as mp
import numpy as np
from math import hypot

# Load the required images
bd_image = cv2.imread("BD.jpeg")
frame = cv2.imread("TestSet_cropped/P1/s01_11.jpg")

height, width, _ = frame.shape
nose_mask = np.zeros((height, width), np.uint8)
nose_mask.fill(0)

# Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()  

# Facial Landmarks
result = face_mesh.process(frame)

for facial_landmarks in result.multi_face_landmarks:

    # Nose Co-ordinates
    top_nose = (facial_landmarks.landmark[8].x * width, facial_landmarks.landmark[8].y * height)
    center_nose = (facial_landmarks.landmark[195].x * width, facial_landmarks.landmark[195].y * height)
    
    left_eye_point = (facial_landmarks.landmark[190].x * width, facial_landmarks.landmark[190].y * height)
    right_eye_point = (facial_landmarks.landmark[413].x * width, facial_landmarks.landmark[413].y * height)


    nose_width = int(hypot(left_eye_point[0] - right_eye_point[0],
                            left_eye_point[1] - right_eye_point[1]))

    nose_width = 50
    nose_height = int(nose_width )   # 0.37 comes from H/W of the Image (373/100) - 1.73 is a random no                       

    top_left = (int(center_nose[0] - nose_width / 2),
                    int(center_nose[1] - nose_height /2))

    bottom_right = (int(center_nose[0] + nose_width / 2),
                        int(center_nose[1] + nose_height / 2))

    # Adding the Band Aid Image
    bd_pic = cv2.resize(bd_image, (nose_width, nose_height))
    bd_pic_gray = cv2.cvtColor(bd_pic, cv2.COLOR_BGR2GRAY)
    _, nose_mask = cv2.threshold(bd_pic_gray, 25, 255, cv2.THRESH_BINARY_INV)

    nose_area = frame[top_left[1]: top_left[1] + nose_height,
                top_left[0]: top_left[0] + nose_width]

    nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)

    final_nose = cv2.add(nose_area_no_nose, bd_pic)

    frame[top_left[1]: top_left[1] + nose_height,
                top_left[0]: top_left[0] + nose_width] = final_nose

    print("Nose Shape - ", final_nose.shape)

    cv2.imshow("Final Nose", final_nose)

cv2.imshow("Frame", frame)
cv2.imwrite("MediaPipe-1x1.png", frame)
key = cv2.waitKey(0)
