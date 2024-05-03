import numpy as np
import cv2
import time
# Load YOLO model
net = cv2.dnn.readNet('yolov3-tiny.cfg', 'yolov3-tiny.weights')  # Update with the pedestrian detection model

# Load COCO labels
with open('coco.names', 'r') as coco:
    classes = coco.read().strip().split('\n')

# Set up video capture
capture = cv2.VideoCapture('/home/aisling/image_recognition_env/test.mov')  

start_time=time.time()
frame_count=0

while True:
    ret, frame = capture.read()
    if not ret:
        break
    

    frame_count+=1

    height, width, _ = frame.shape

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get output layer names
    output_layer_names = net.getUnconnectedOutLayersNames()

    # Run forward pass
    outs = net.forward(output_layer_names)

    # Loop over each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] == 'person':
                # Object detected
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(int(detection[0] * width) - w / 2)
                y = int(int(detection[1] * height) - h / 2)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, classes[class_id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(33) == 27:
       break

end_time=time.time() 
# Calculate frames per second
fps=frame_count/(end_time-start_time)
print(frame_count)
print(end_time-start_time)
print(fps)

# Release the video capture object and close all windows
capture.release()
cv2.destroyAllWindows()