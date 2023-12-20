# WELCOME TO OUR 255217 PRODUCTION SUPPORT SYSTEM IN FACTORIES PROJECT!

This project involves the implementation of a Face Recognition-based Attendance Checking System, integrated with a Node-Red Flow.

The Face Recognition Program employs OpenCV and DeepFace to identify faces and emotions through an internal or streaming camera, delivering the output via MQTT.

Node-Red flow have 4 sections:
1. Monitor : Uses data from Face Reconition Program via MQTT and location data from mobile application "owntrack" via MQTT then check whether the user is in designated perimeter or not and process the attendace criteria.
2. Notification: Ater the data has been added to Firestore it will output to Line Notification API.
3. Control: Operators can control Attendace system by manual button via Node-Red dashboard or timer
4. Storage: Node-Red uses Firestore to store all the data and csv in local storage.

**Face Recognition Program Instruction**

1. Config camera and MQTT server, topic in config.yaml.
2. Run dataset.py to gather face data for training.
3. Run train.py to train recognizer.
4. Run main.py for main program.

Credits for:
1.ajitharunai/Facial-Emotion-Recognition-with-OpenCV-and-Deepface/
2.Mjrovai/OpenCV-Face-Recognition
