import cv2
import mediapipe as mp

image = cv2.imread("Rushi.png")
image = cv2.resize(image, (1024, 1024))

# Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()  

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Facial Landmarks
result = face_mesh.process(rgb_image)

height, width, _ = image.shape

for facial_landmarks in result.multi_face_landmarks:
    for i in range(0, 468):
        point = facial_landmarks.landmark[i]  # This gives us percentage position of the object

        x = int(point.x * width)           # Multiply the percentage with actual widht and height
        y = int(point.y * height)

        cv2.circle(image, (x, y), 1, (100, 100, 0), -1)
        cv2.putText(image, str(i), (x, y), 0, 0.4, (0, 0, 0))

cv2.imshow("Image", image)
cv2.imwrite("468Points.png", image)

cv2.waitKey(0)