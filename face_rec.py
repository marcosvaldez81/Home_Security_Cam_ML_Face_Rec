import cv2
import os
import numpy as np
import pickle
import boto3
import Secrets import access_key, secret_access_key

face_cascade=cv2.CascadeClassifier('/home/pi/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('/home/pi/opencv/data/haarcascades/haarcascade_eye.xml')
smile_cascade=cv2.CascadeClassifier('/home/pi/opencv/data/haarcascades/haarcascade_smile.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels={}

with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}


cap = cv2.VideoCapture(0)
num = 0
while(True):
    # capture frame-by-frame
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5,minSize=(20, 20))

    for(x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]


        #recognize? deep learned model predict keras keras, tensorflow pytorch...
        id_,conf = recognizer.predict(roi_gray)
        if conf >= 45: # and conf <= 85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame,name,(x,y), font, 1, color, stroke, cv2.LINE_AA)


        #img_item = "7.png"
        #cv2.imwrite(img_item, frame)
        #num= num+ 1

        color = (255,0,0) # not an RGB like HTML, it is BGR 0-255

        stroke = 4
        end_cord_x = x+w
        end_cord_y = y+h
        cv2.rectangle(frame, (x,y), (end_cord_x,end_cord_y), color, stroke)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey), (ex+ew, ey+eh), (0,255,0),2)

        smiles = smile_cascade.detectMultiScale(roi_gray,
             scaleFactor= 1.5,
             minNeighbors=15,
             minSize=(25, 25),)
        for (xx, yy, ww, hh) in smiles:
             cv2.rectangle(roi_color, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)
        img_item = (str(num) + ".png")
        cv2.imwrite(img_item, frame)
        num= num+ 1


    #Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    elif num >=150000: # Take 30 face sample and stop video
        break

client = boto3.client('s3',
                          aws_access_key_id = access_key,
                          aws_secret_access_key = secret_access_key)


for file in os.listdir():
    if '.png' in file:
        upload_file_bucket = 'pisecuritycam'
        upload_file_key = 'images/' + str(file)
        client.upload_file(file, upload_file_bucket,upload_file_key)
        print("Done!")



cap.release()
cv2.destroyAllWindows
