import cv2
import numpy as np
import time
import threading

cv2.startWindowThread()
#capture = cv2.VideoCapture('/home/aisling/image_recognition_env/dashCamFootpath.mp4')
#capture = cv2.VideoCapture('/home/aisling/image_recognition_env/dashCamTownCentre.mp4')
capture = cv2.VideoCapture(0)
# Load in tiny YOLO
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
# Specifies classes for detection
with open('coco.names', 'r') as coco:
    classes = coco.read().strip().split('\n')

# Get associated classes for output
layer_names = net.getLayerNames()
unconnected_layers = net.getUnconnectedOutLayers()
if isinstance(unconnected_layers, int):
   unconnected_layers = [unconnected_layers]
output_layers = [layer_names[i - 1] for i in unconnected_layers]

output_video = cv2.VideoWriter('finalOutput6.mp4', cv2.VideoWriter_fourcc(*'mp4v'),8, (640, 200))

def non_max_suppression(detections, threshold):

    detections = sorted(detections, reverse=True)
    nms_detcetions = []

    while len(detections) > 0:
        nms_detcetions.append(detections[0])
        del detections[0]

        i = 0
        while i < len(detections):
            # Specifies the bounds of the intersection
            x1 = max(nms_detcetions[-1][0], detections[i][0])
            y1 = max(nms_detcetions[-1][1], detections[i][1])
            x2 = min(nms_detcetions[-1][0] + nms_detcetions[-1][2], detections[i][0] + detections[i][2])
            y2 = min(nms_detcetions[-1][1] + nms_detcetions[-1][3], detections[i][1] + detections[i][3])

            intersection_area = (x2-x1)*(y2-y1)
            nms_area = nms_detcetions[-1][2] * nms_detcetions[-1][3]
            detection_area = detections[i][2] * detections[i][3]
            union_area=nms_area+detection_area-intersection_area
            IoU = intersection_area / union_area

            if IoU > threshold:
                del detections[i]
            else:
                i += 1

    return nms_detcetions

def process_frames(capture):
    frame_count = 0
    start_time = time.time()

    while True:
        # Capture frame-by-frame
        ret, frame = capture.read()
        if not ret:
            break
        
        frame_count += 1

        if frame_count % 5 == 0: #process 1 in 3 frames
            # Resizing for faster detection
            resize_frame = cv2.resize(frame, (640, 480))
            # Dimensions of specific area to analysis
            #x,y,w,h,=(747,224,416,416)
            #x,y,w,h,=(0,0,640,480)
            x,y,w,h,=(0,100,640,200)
            # Applying dimensions to frame
            analysis_frame=resize_frame[y:y+h,x:x+w]

            height, width, channels = analysis_frame.shape
            # Apply tiny YOLO algorithm detection
            blob = cv2.dnn.blobFromImage(analysis_frame, 1/255, (640,210), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            output = net.forward(output_layers)
            
            detections = []
            # Creates detection rectangles
            for out in output:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(int(detection[0] * width) - w / 2)
                        y = int(int(detection[1] * height) - h / 2)
                        detections.append((x, y, w, h, class_id))
            
            detections = non_max_suppression(detections, 0.2)

            # Draw detection rectangle with object name
            for x, y, w, h, class_id in detections:
                cv2.rectangle(analysis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText( analysis_frame, classes[class_id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            output_video.write(analysis_frame)

            # Displays the frame with detection rectangles
            cv2.imshow('Object Detection',  analysis_frame)

            # Escapes the livestream if the Esc key is pressed
            if cv2.waitKey(33) == 27:
                break

    end_time = time.time()
    # Calculate frames per second
    fps = frame_count / (end_time - start_time)
    print("Frames processed:", frame_count)
    print("Time taken:", end_time - start_time)
    print("FPS:", fps)
    output_video.release()


# Sets up thread for processing frames
#thread = threading.Thread(target=process_frames, args=(capture,))
#thread.start()
# Waits for thread to finish
#thread.join()
process_frames(capture)
# Ends capture and closes display
capture.release()
cv2.destroyAllWindows()



