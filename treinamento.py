import PIL
import cv2
import os
from cv2 import threshold
import numpy as np
from PIL import Image


eigenFace = cv2.face.EigenFaceRecognizer_create(threshold= 3)
fisherFace = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagemComId(path):
    width_d, height_d = 280, 280
    caminhos = [os.path.join('images', f) for f in os.listdir('images')]
    faceSamples=[]
    Ids=[]
    i = 0
    for imagePath in caminhos:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage,'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(imageNp)
        for (x,y,w,h) in faces:
            faceSamples.append(cv2.resize(imageNp[y:y+h,x:x+w], (width_d, height_d)))
            Ids.append(Id)
    return faceSamples,Ids

faces,Ids = getImagemComId('images')
print("Treinamento...")

eigenFace.train(faces, np.array(Ids))
eigenFace.write('classificadorEigen.yml')

lbph.train(faces, np.array(Ids))
lbph.write('classificadorLBPH.yml')

fisherFace.train(faces, np.array(Ids))
fisherFace.write('classificadorFisher.yml')

print("Treinamento realizada")



