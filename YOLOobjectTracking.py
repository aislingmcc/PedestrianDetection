import cv2 
import numpy as np 
import time 

cv2.startWindowThread()

# Load YOLO model
net = cv2.dnn.readNet("/home/pi/image_recognition_env/darknet/yolov3-tiny.weights", "/home/pi/image_recognition_env/darknet/cfg/yolov3-tiny.cfg")

# Load class labels
classes = []
with open("/home/pi/image_recognition_env/darknet/data/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize trackers and IDs
trackers = []
tracker_ids = []
next_id = 1

# Function to update trackers on each frame
def update_trackers(frame):
    global trackers, tracker_ids, next_id
    new_trackers = []
    new_tracker_ids = []

    for tracker, tracker_id in zip(trackers, tracker_ids):
        # Update tracker with the current frame
        success, box = tracker.update(frame)
        if success:
            # If the update is successful, update tracker lists and draw bounding box
            new_trackers.append(tracker)
            new_tracker_ids.append(tracker_id)
            x, y, w, h = [int(v) for v in box]

            # Output of active trackers and data is continuously updated
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Tracker ID: {tracker_id}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"[ID: #{tracker_id}, ({x}, {y})]")
        else:
            # If the update fails, print a message
            print(f"[ID: #{tracker_id}] - Update failed")
    return new_trackers, new_tracker_ids

# Get YOLO output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
# Open video capture device
capture = cv2.VideoCapture('/home/pi/image_recognition_env/dashCamTownCentre.mp4')
start_time = time.time()
frame_count = 0

while True:
    ret, frame = capture.read()
    
    if not ret:
        break

    # Resize frame to a smaller size for faster processing
    frame = cv2.resize(frame, (960, 540)) # Example resizing to 640x360, adjust as needed
    frame_count += 1
    trackers, tracker_ids = update_trackers(frame)

    #Original frame dimensions replaced by resized frame dimensions
    height, width, channels = frame.shape

    if frame_count % 5 == 0:
        # Create blob from frame for YOLO input
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        boxes = []
        confidences = []
        class_ids = []

        # Process YOLO output
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Filters out detected bounding boxes that are no longer within the frame
                    if x >= 0 and y >= 0 and x + w <= width and y + h <= height:
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

        # Apply NMS to YOLO detections
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.7, nms_threshold=0.01)

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]

                # Initialize a tracker for each detected object
                tracker = cv2.legacy.TrackerMOSSE_create()
                tracker.init(frame, (x, y, w, h))
                trackers.append(tracker)
                tracker_ids.append(next_id)
                next_id += 1

                # Draw bounding box and ID on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Tracker ID: {next_id}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display frame with detected objects
    cv2.imshow('Object Detection', frame)

    # Check for user input to exit the loop
    if cv2.waitKey(33) == 27:
        break

# Calculate and print frame processing statistics
end_time = time.time()
fps = frame_count / (end_time - start_time)
print("Frame Count:", frame_count)
print("Total Time:", end_time - start_time)
print("FPS:", fps)

# Release video capture device and close all OpenCV windows
capture.release()
cv2.destroyAllWindows()
 