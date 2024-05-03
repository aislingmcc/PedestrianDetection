#HOGimprovedPedestrianDetection
import cv2
import time

# Initializing the HOG person  
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
video = '/home/aisling/image_recognition_env/dashCam.mp4' 

# Code to implement livestream detection on a usb camera  
cv2.startWindowThread()

# Comment in the following to detect from livestream  
capture = cv2.VideoCapture(0)

# Comment in the following to detect from video  
#capture = cv2.VideoCapture(video)  

# Count to control frames analysed  
count=5

start_time=time.time()
frame_count=0

while(True):
    # Capture frame-by-frame  
    ret, frame = capture.read()

    if not ret:
        break

    count+=1 
    frame_count+=1

    # Resizing for faster detection  
    resize_frame = cv2.resize(frame, (640, 480))

    # Dimensions of specific area to analysis  
    x,y,w,h,=(0,120,640,190)

    # Applying dimensions to frame  
    analysis_frame=resize_frame[y:y+h,x:x+w]

    if count >5: #process 1 in 5  
        count = 0

        # using a greyscale picture, also for faster detection  
        gray = cv2.cvtColor(analysis_frame, cv2.COLOR_RGB2GRAY)
        # detect people in the image  
        # returns the bounding boxes for the detected objects  
        region, weights = hog.detectMultiScale(gray,winStride=(4,4),padding=(8,8),scale=1.05)

    # Drawing the regions in the frame  
    for (x, y, w, h) in region:cv2.rectangle(analysis_frame,(x, y),(x + w, y + h),(0, 0, 255), 2)

    # Displays the output stream  
    cv2.imshow("Livestream", analysis_frame)

    # Escapes the livestream if the Esc key is pressed 
    if cv2.waitKey(33) == 27:
        break

end_time=time.time()

# Calculate frames per second  
fps=frame_count/(end_time-start_time)
print(frame_count)
print(end_time-start_time)
print(fps)

# Ends caputre and closes display  
capture.release()
cv2.destroyAllWindows()