import cv2
import argparse
import numpy as np
from scipy.spatial import distance as dist

parser = argparse.ArgumentParser(
    description='Script to run MobileNet-SSD object detection network ')
parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt",
                                  help='Path to text network file: '
                                       'MobileNetSSD_deploy.prototxt for Caffe model or '
                                       )
parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel",
                                 help='Path to weights: '
                                      'MobileNetSSD_deploy.caffemodel for Caffe model or '
                                      )
parser.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
args = parser.parse_args()

#Labels of network.
classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }


# Open video file or capture device.
if args.video:
    cap = cv2.VideoCapture(args.video)
else:
    cap = cv2.VideoCapture(0)

#Load the Caffe model
net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_resized = cv2.resize(frame,(300,300)) # resize frame for prediction

    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    net.setInput(blob)
    detections = net.forward()

    # Size of frame resize (300x300)
    cols = frame_resized.shape[1]
    rows = frame_resized.shape[0]
    centroids = []
    boxes = []

    MIN_DISTANCE = 70
    human_detected = True
    violation = set()
    # For get the class and location of object detected,
    # There is a fix index for class, location and confidence
    # value in @detections array .
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Confidence of prediction
        if confidence > args.thr:  # Filter prediction
            class_id = int(detections[0, 0, i, 1])  # Class label
            if class_id != 15: # Person not detected
                human_detected = False
                continue
            human_detected = True
            # Object location
            xLeftBottom = int(detections[0, 0, i, 3] * cols)
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop = int(detections[0, 0, i, 5] * cols)
            yRightTop = int(detections[0, 0, i, 6] * rows)

            # Factor for scale to original size of frame
            heightFactor = frame.shape[0] / 300.0
            widthFactor = frame.shape[1] / 300.0
            # Scale object detection to frame
            xLeftBottom = int(widthFactor * xLeftBottom)
            yLeftBottom = int(heightFactor * yLeftBottom)
            xRightTop = int(widthFactor * xRightTop)
            yRightTop = int(heightFactor * yRightTop)
            # Draw location of object
            x_middle = int((xRightTop+xLeftBottom) /2)
            y_middle = int((yRightTop+yLeftBottom)/2)

            centroids.append((x_middle,y_middle))
            boxes.append((xLeftBottom,yLeftBottom,xRightTop,yRightTop))
            img_center = x_middle,y_middle
            # cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
            #               (0, 255, 0),2)
            cv2.circle(frame,img_center,radius=1,color=(0, 0, 255))

            height = round(yLeftBottom - yRightTop,4)

            # Draw label and confidence of prediction in frame resized
            if class_id in classNames:
                label = classNames[class_id] + ": " + str(confidence)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                yLeftBottom = max(yLeftBottom, labelSize[1])
                cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                              (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                              (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                print(label)  # print class and confidence
        if human_detected:
            D = dist.cdist(centroids,centroids, metric="euclidean")

            for k in range(0, D.shape[0]):
                for j in range(k + 1, D.shape[1]):
                    # check to see if the distance between any two
                    # centroid pairs is less than the configured number
                    # of pixels
                    print("distance")
                    print(D[k, j])

                    print(D[k, j] < MIN_DISTANCE)
                    if D[k, j] < MIN_DISTANCE:
                        # update our violation set with the indexes of
                        # the centroid pairs
                        violation.add(k)
                        violation.add(j)

            for z in range(0,len(centroids)):
                color = (0, 255, 0)

                if z in violation:
                    color = (0, 0, 255)

                cv2.rectangle(frame, (boxes[z][0], boxes[z][1]), (boxes[z][2], boxes[z][3]),
                              color,2)


        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) >= 0:  # Break with ESC
            break