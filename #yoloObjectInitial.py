#yoloObjectInitial
import cv2
import numpy as np
import time
print(cv2.__version__)

capture = cv2.VideoCapture('/home/aisling/image_recognition_env/dashCamTownCentre.mp4') 
# Load in tiny YOLO
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")

# Specifies classes for detection
with open("coco.names", "r") as f:
   classes = [line.strip() for line in f.readlines()]

# Get associated classes for output
layer_names = net.getLayerNames()
unconnected_layers = net.getUnconnectedOutLayers()
if isinstance(unconnected_layers, int):
   unconnected_layers = [unconnected_layers]
output_layers = [layer_names[i - 1] for i in unconnected_layers]

start_time=time.time()
frame_count=0

while True:
    ret, frame = capture.read()
    if not ret:
       break

    frame_count+=1
    height, width, channels = frame.shape

    # Apply tiny YOLO algorithm detection
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    detections=[]
    # Creates detection rectangles
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(int(detection[0] * width) - w / 2)
                y = int(int(detection[1] * height) - h / 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, classes[class_id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                
    # Displays the frame with detection rectangles
    cv2.imshow('Object Detection', frame)
    
    # Escapes the livestream if the Esc key is pressed
    if cv2.waitKey(33) == 27:
        break

end_time=time.time()
# Calculate frames per second
fps=frame_count/(end_time-start_time)
print(frame_count)
print(end_time-start_time)
print(fps)

# Ends capture and closes display
capture.release()
cv2.destroyAllWindows()
