#HaarCascade
import cv2
import time 

# Path to the Haar cascade for pedestrian detection  
haar = '/home/aisling/image_recognition_env/haarcascade_fullbody.xml'
# Path to the video file  
video = '/home/aisling/image_recognition_env/dashCam.mp4'
# Initialise VideoCapture and classifier  
capture = cv2.VideoCapture(video)
fullbody_cascade = cv2.CascadeClassifier(haar)
# Initialise start time and frame count for frame processing rate  
start_time=time.time()
frame_count=0
# Loop through video  
while True:
    # Read frames from the video  
    ret, frame = capture.read()
    # Ends loop when video ends  
    if not ret:
        break

    frame_count+=1
    # Resizing and adjusting to gray scale will speed up the detection  
    resize_frame=cv2.resize(frame,(640,480))
    gray = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2GRAY)
    # Detect pedestrians in the grayscale, resized frame  
    pedestrian = fullbody_cascade.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=3,minSize=(30, 70)) 

    # Draw rectangles around detected pedestrians  
    for (x, y, w, h) in pedestrian:cv2.rectangle(resize_frame, (x, y),(x+w, y+h), (0, 255, 0), 2)
    # Displays video with detection rectangles  
    cv2.imshow('Video', resize_frame)

    # Breaks the loop if the 'Esc' key is pressed  
    if cv2.waitKey(33) == 27:
        break

end_time=time.time()

# Calculate frames per second  
fps=frame_count/(end_time-start_time)
print(frame_count)
print(end_time-start_time)
print(fps)

# Ends caputre and closes displayed video  
capture.release()
cv2.destroyAllWindows()