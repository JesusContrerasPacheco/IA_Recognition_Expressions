import os
import numpy as np
import cv2 as cv2
import glob
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import pickle
import pandas as pd

data = { 'USER': [], 'PREDICT': [] }

path = 'D:/UPC/2021 - 1/IA/proyecto/IA_Recognition_Expressions/'
path_img = 'D:/UPC/2021 - 1/IA/proyecto/IA_Recognition_Expressions/testImages/'
#rutimg = str(input("Coloca la ruta de la imagen:"))
files_names = os.listdir(path_img)
#print(files_names)


# def returnLocalizedFace(img):
#     face_cascade = cv2.CascadeClassifier(path + 'classifiers/haarcascade_frontalface_default.xml')
#     gray =cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     for (x, y, w, h) in faces:
#         # roi_gray = gray[y:y + h, x:x + w]
#         # roi_color = img[y:y + h, x:x + w]
#         if len(faces) == 0:
#             return img
#         crop_img = img[y:y + h, x:x + w]
#         return crop_img


def getImage(path_img):
    return cv2.imread(path_img)
#   for files_name in files_names:
#        image = cv2.imread(path_img +"/"+files_name)
#        return image    
        
#print(getImage(path_img))

def show(img):
    cv2.imshow('im', img)
    # cv2.waitKey(0)

X = []
y = []

def read(imageFolder, label):

    for filename in glob.glob(path + "images/" + imageFolder + '/*.*'):
        win_size = (64, 128)
        img = getImage(filename)

        win_size = (64, 128)

        img = cv2.resize(img, win_size)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        d = cv2.HOGDescriptor()
        hog = d.compute(img)
        X.append(hog.transpose()[0])
        y.append(label)

# def fromIndexToFeatures(X, indecies):
#     features = []ads
#     for i in indecies:
#         features.append(X[i])
#     return np.asarray(features)

# def fromIndexToLabels(y, indecies):
#     labels = []
#     for i in indecies:
#         labels.append(y[i])
#     return np.asarray(labels)

def openModal(img):
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(path + "classifiers/haarcascade_frontalface_alt2.xml")
    faces = face_cascade.detectMultiScale(image_gray)
    landmark_detector  = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel(path + "classifiers/LFBmodel.yaml")
    _, landmarks = landmark_detector.fit(image_gray, faces)
    for landmark in landmarks:
        for x,y in landmark[0]:
            cv2.circle(img, (int(x), int(y)), 1, (255, 255, 255), 1)
    
   # show(img)

read('HAPPY',0)
read('CONTEMPT',1)
read('ANGER',2)
read('DISGUST',3)
read('FEAR',4)
read('SADNESS',5)
read('SURPRISE',6)
read('NEUTRAL',7)

classes = ["HAPPY", "CONTEMPT", "ANGER", "DISGUSTADO", "FEAR", "SADNESS", "SURPRISE", "NEUTRAL"]
y = np.asarray(y)
X = np.asarray(X)

clf = OneVsRestClassifier(SVC(kernel='linear', probability=True, tol=1e-3))
clf.fit(X, y)

filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))

clf = pickle.load(open(filename, 'rb')) 

#img = getImage(path_img)
#path + 'testImages/angry.jpg'
# points 
#print (img)

win_size = (64, 128)
"""
img = cv2.resize(img, win_size)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
d = cv2.HOGDescriptor()
hog = d.compute(img)

hog = hog.transpose()[0]
hog = np.asarray(hog)
print(hog)
"""
for files_name in files_names:
    #print(files_name)

    data["USER"].append(files_name)

    image_path = path_img + files_name
    img = getImage(image_path)
    if img is None:
        continue
    # openModal(img)
    # show(img)
    img = cv2.resize(img, win_size)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    d = cv2.HOGDescriptor()
    hog = d.compute(img)

    hog = hog.transpose()[0]
    hog = np.asarray(hog)
    print(hog)

    result = classes[clf.predict([hog])[0]]

    print("Result: " + result)

    data["PREDICT"].append(result)

    cv2.waitKey(0)
cv2.destroyAllWindows()

df = pd.DataFrame(data, columns = ['USER', 'Nivel Sentimiento(+)' ,'Nivel Sentimiento(-)', 'PREDICT'])
df.to_excel(path + "export.xlsx", index = False, header=True)