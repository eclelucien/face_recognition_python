import cv2

classificador = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
camera = cv2.VideoCapture(0)

while(True):
    conectado, image = camera.read()
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    facesDetectadas = classificador.detectMultiScale(grayImage, scaleFactor=1.5, minSize=(100,100))
    
    for(x, y, l, a) in facesDetectadas:
        cv2.rectangle(image, (x, y), (x + l, y + a), (0,0, 255), 2)
    
    cv2.imshow("Face", image)
    cv2.waitKey(1)

camera.release()
cv2.destroyAllWindows()