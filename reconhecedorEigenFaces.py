import os
import cv2
import numpy as np
from PIL import Image


detectorFace= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
reconhecedor = cv2.face.EigenFaceRecognizer_create()   
reconhecedor.read("classificadorEigen.yml")
font = cv2.AKAZE_DESCRIPTOR_KAZE_UPRIGHT
largura, altura = 280, 280
caminhos = [os.path.join('imageTest', f) for f in os.listdir('imageTest')]

for imagePath in caminhos:
    pilImage = Image.open(imagePath).convert('L')
    imageNp = np.array(pilImage,'uint8')
    faces = detectorFace.detectMultiScale(imageNp)
    for (x,y,w,h) in faces:
        id, confianca  = reconhecedor.predict(cv2.resize(imageNp[y:y+h,x:x+w], (largura, altura)))
        name = "?????"
        if(id == 1):
            name = "Eclesiaste Lucien"
        elif(id == 2):
           name = "Cristiano Ronaldo"
        elif(id == 3):
           name = "Mario Balotelli"
        cv2.putText(imageNp, str(confianca),   (x, y + (h + 50)), font, 1, (0, 0, 255))
        cv2.putText(imageNp, name,   (x, y + (h + 10)), font, 3, (0, 0, 255))
    cv2.imshow("Face", imageNp)
    cv2.waitKey(0)