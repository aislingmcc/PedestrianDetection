#HOGinitialPedestrianDetection
import cv2 
import time

# Initialising the HOG descriptor  
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
capture = cv2.VideoCapture('/home/aisling/image_recognition_env/dashCam.mp4')

# Initialise start time and frame count for frame processing rate  
start_time=time.time()
frame_count=0

# Loop through video  
while(True):
    # Capture frame  
    ret, frame = capture.read()
    # End loop when video is complete  
    if not ret:
        break

    frame_count+=1

    # Resize frame and use grayscale for faster detection  
    resize_frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(resize_frame, cv2.COLOR_RGB2GRAY)

    # Detects pedestrians in each frame following guidelines of parameters  
    pedestrian, weights = hog.detectMultiScale(gray, winStride=(4,4),padding=(4,4),scale=1.1)

    # Drawing the regions a pedestrian is present  
    for (x, y, w, h) in pedestrian:cv2.rectangle(resize_frame, (x, y),(x + w, y + h),(0, 0, 255), 2)

    # Display the output video  
    cv2.imshow("video", resize_frame)

    # Escapes if the Esc key is pressed  
    if cv2.waitKey(33) == 27:
        break


end_time=time.time() 

# Calculates the frames per second  
fps=frame_count/(end_time-start_time)
print(frame_count)
print(end_time-start_time)
print(fps)

capture.release()
cv2.destroyAllWindows()