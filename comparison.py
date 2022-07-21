import face_recognition
from PIL import Image


# def getFaceInPicture(image1, image2):
#     firstPictureToCompare = face_recognition.load_image_file("images/" + image1)
#     my_face_encoding = face_recognition.face_encodings(firstPictureToCompare)[0]
    
#     secondPictureToCompare = face_recognition.load_image_file("images/" + image2)
#     unknown_face_encoding = face_recognition.face_encodings(secondPictureToCompare)[0]
    
#     results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)

#     if results[0] == True:
#         print("Yes this is the same person in this picure")
#     else:
#         print("No, there are two diffrent persons in this picture")

# getFaceInPicture(image1= "face.1.4.jpeg", image2 = "face.3.1.jpg")






# Load the jpg files into numpy arrays
biden_image = face_recognition.load_image_file("othersImages/mane.jpg")
obama_image = face_recognition.load_image_file("othersImages/mbappe.jpeg")
unknown_image = face_recognition.load_image_file("othersImages/mane.jpg")

# Get the face encodings for each face in each image file
# Since there could be more than one face in each image, it returns a list of encodings.
# But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
try:
    biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
    unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
except IndexError:
    print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    quit()

known_faces = [
    biden_face_encoding,
    obama_face_encoding
]

# results is an array of True/False telling if the unknown face matched anyone in the known_faces array
results = face_recognition.compare_faces(known_faces, unknown_face_encoding)

print("Is the unknown face a picture of Biden? {}".format(results[0]))
print("Is the unknown face a picture of Obama? {}".format(results[1]))
print("Is the unknown face a new person that we've never seen before? {}".format(not True in results))