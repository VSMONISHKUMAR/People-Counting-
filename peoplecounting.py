import cv2
import numpy as np


net = cv2.dnn.readNet("yolov2.weights", "yolov2.cfg")
layer_names = net.getUnconnectedOutLayersNames()


cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()

   
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

   
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0: 
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

   
    count = len(indices)
    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    
    cv2.putText(frame, f"People Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("People Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
