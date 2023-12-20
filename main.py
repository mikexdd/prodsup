import cv2
import numpy as np
import os 
import imutils
import requests
import paho.mqtt.client as mqtt
import yaml
import json
import csv
from deepface import DeepFace

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('data/trainer/trainer.yml')
cascadePath = "data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX

output = {"id":0,"emotion":"none"}

#Config Reader
with open("config.yaml", "r") as config_file:
  config = yaml.safe_load(config_file)

#People Data Reader
names = []
with open('data/people.csv', mode='r') as csv_file:
  name_csv = csv.reader(csv_file, delimiter=",")
  line_count = 0
  for row in name_csv:
    if line_count == 0:
        line_count += 1
    else:
        names.append(row[1])
        line_count += 1
      

#iniciate id counter
id = 0

# Initialize and start realtime video capture
cam = cv2.VideoCapture(config["cam_source"])
cam.set(3, 960) # set video widht
cam.set(4, 720) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

# Face Export
latest_scan = "None"

#MQTT
host = config["mqtt"]["host"]
port = config["mqtt"]["port"]
def on_connect(self, client, userdata, rc):
    print("MQTT Connected.")
    self.subscribe("prodsup/mqtt")
def on_message(client, userdata,msg):
    print(msg.payload.decode("utf-8", "strict"))
client = mqtt.Client()
client.connect(host)

#Emotion Recognition
model = DeepFace.build_model("Emotion")
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


while True:
    
    ret, img = cam.read()
    #img = cv2.flip(img, -1) # Flip vertically    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:
        w2 = int(w/2)
        # Emotion Recognition
        # Extract the face ROI (Region of Interest)
        face_roi = gray[y:y + h, x:x + w]

        # Resize the face ROI to match the input shape of the model
        resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)

        # Normalize the resized face image
        normalized_face = resized_face / 255.0

        # Reshape the image to match the input shape of the model
        reshaped_face = normalized_face.reshape(1, 48, 48, 1)

        # Predict emotions using the pre-trained model
        preds = model.predict(reshaped_face)[0]
        emotion_idx = preds.argmax()
        emotion = emotion_labels[emotion_idx]

        output["emotion"] = emotion
        # Face Recognition
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            output["id"] = id
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
            scan_match = id
            if scan_match == 0:
                scan_match += 1
            if latest_scan == id :
                print("Scanned")
            else:
                latest_scan = id
                print(latest_scan)
                client.publish(config["mqtt"]["output_topic"],str(json.dumps(output)))

        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        # Emotion Display
        cv2.putText(img, emotion, (x+w2, y-5), font, 1, (0, 0, 255), 2)
        # Face id Display
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('Super Cam!',img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()