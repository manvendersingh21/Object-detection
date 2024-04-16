import cv2
import numpy as np
from ultralytics import YOLO


cap= cv2.VideoCapture("dogs.mp4")

# object detection mode
model = YOLO("yolov8m.pt")
data=open("classes.txt","r")
lines=data.readlines()
data.close()



while True:
    #"ret" basically meany if we have a frame or not "True" or "false"
    #saves the frame in one code
    ret, frame = cap.read()

    if not ret:
        break

    results= model(frame, device="mps")
    #print(results) pi
    result=results[0]
    bboxes=np.array(result.boxes.xyxy.cpu(),dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")

    for cls,bbox in zip(classes,bboxes):
        (x,y,x2,y2)=bbox
        
        cv2.rectangle(frame,(x,y),(x2,y2),(0,0,225),2)
        cv2.putText(frame,str(cls),(x,y-5),cv2.FONT_HERSHEY_PLAIN, 4 ,(0,0,225),5)
        cv2.putText(frame,str(lines[cls])[:-1],(x2,y-5),cv2.FONT_HERSHEY_PLAIN, 4 ,(0,0,225),5)

    #"img" means picture in frame whch is showed
    cv2.imshow("img", frame)

    #"1" means wait for 1 millisecind then show different frame "0" means wait till something is pressed
    key = cv2.waitKey(0)

    if key==27:
        break

#it releases the video " so that no error pops up of holding the video"& cv2 to close all the windows opened by us
cap.release()
cv2.destroyAllWindows
