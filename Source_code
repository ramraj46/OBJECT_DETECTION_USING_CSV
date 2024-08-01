# Importing the csv module for detection and numpy module to represent the output
import cv2
import numpy as np
# config_file contains the configuration of the deep learning model
config_file = r"C:\Users\ramar\Downloads\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
# frozen_model contains the trained weights and network architecture
frozen_model = r"C:\Users\ramar\Downloads\frozen_inference_graph.pb"

model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = []
# filename consists of the trained objects of the model
filename = r"C:\Users\ramar\Downloads\labels.txt"
with open(filename, 'rt') as spt:
    classLabels = spt.read().rstrip('\n').split('\n')
# Greater this value better the results, tune it for best output
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

img = cv2.imread(r"C:\Users\ramar\Downloads\t1.jpeg")

classIndex, confidence, bbox = model.detect(img, confThreshold=0.5) # Tune confThreshold for best results

font = cv2.FONT_HERSHEY_PLAIN

for classInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
    cv2.rectangle(img, boxes, (255, 0, 0), 2)
    cv2.putText(img, classLabels[classInd - 1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=3, color=(0, 255, 0), thickness=3)

# Show the result
cv2.imshow('result', dimmed_img)
# this is part is for quiting the program
# Wait for a key press indefinitely
key = cv2.waitKey(0)

# If 'q' is pressed, exit
if key == ord('q'):
    cv2.destroyAllWindows()  # Close the image window
