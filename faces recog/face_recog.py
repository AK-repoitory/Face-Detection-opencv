
import cv2
import os
import numpy as np
import pickle
from PIL import Image
# cretae a folder train to store data set .
# cascades = https://github.com/Itseez/opencv/tree/master/data/haarcascades

def recognize():
    cam = cv2.VideoCapture(0)
    reccognizer = cv2.face.LBPHFaceRecognizer_create()
    face_haar_cascade = cv2.CascadeClassifier(
        '/Users/DELL/Desktop/faces recog/cascades/data/haarcascade_frontalface_default.xml')
    reccognizer.read('/Users/DELL/Desktop/faces recog/trainner.yaml')

    final = []
    lable = {}
    no = 0
    with open("/Users/DELL/Desktop/faces recog/labes.pickle", "rb") as f:
        lables = pickle.load(f)
        print(lables)
        lable = {v: k for k, v in lables.items()}

    while True:
        ret, img = cam.read()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.5, minNeighbors=5)
        for (x, y, w, h) in faces:
            roi_gray = gray_img[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            id_, conf = reccognizer.predict(roi_gray)
            if conf >= 50:  # and conf <=85:
                final.append(id_)
                # print(id_)
        if cv2.waitKey(2) == ord('q'):
            break
        for face in faces:
            (x, y, w, h) = face

            if face.all() != 0:
                no = no + 1
                print(1)

        if no == 20:
            break

    cam.release()
    cv2.destroyAllWindows()
    print(final)
    m = max(final)
    
    if m is not None:
        l = final.count(m)
        if l >= 18:
            print(m)
            print(lable)
     


def face_Detection(test_img):
    gray_img =cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    face_haar_cascade =cv2.CascadeClassifier('/Users/DELL/Desktop/faces recog/cascades/data/haarcascade_frontalface_default.xml')
    face = face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.5,minNeighbors=5)
    return face, gray_img


def entry(name):
    cap = cv2.VideoCapture(0)
    count = 0
    i = 0
    no = 0
    f = 0

    os.chdir('/Users/DELL/Desktop/faces recog/train')
    os.mkdir(name)
    os.chdir('/Users/DELL/Desktop/faces recog')

    while True:
        ret, test_img = cap.read()
        if not ret:
            continue
        face_detected, gray_img = face_Detection(test_img)
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        face_haar_cascade = cv2.CascadeClassifier(
            '/Users/DELL/Desktop/faces recog/cascades/data/haarcascade_frontalface_default.xml')
        face = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.5, minNeighbors=5)


        for face in face_detected:
            (x, y, w, h) = face

            if face.all() != 0:
                f = 1
                no = no + 1
                # print(no)
            else:
                f = 0

        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('testing', resized_img)
        if cv2.waitKey(10) == ord('q'):
            break
        if f == 1:
            cv2.imwrite("train/" + name + "/frame%d.jpg" % count, test_img)
            count += 1
            f = 0
        if no == 45:
            break

    cap.release()
    cv2.destroyAllWindows()
    os.chdir('/Users/DELL/Desktop/faces recog')

# labeling the data set

def lable():

    y_lable = []
    x_train = []

    face_haar_cascade = cv2.CascadeClassifier('/Users/DELL/Desktop/faces recog/cascades/data/haarcascade_frontalface_default.xml')
    reccognizer = cv2.face.LBPHFaceRecognizer_create()
    bace_dir = os.path.dirname(os.path.abspath((__file__)))
    imd_dir = os.path.join(bace_dir, "train")
    current_id = 0
    lable_id = {}
    for root, dir, files in os.walk(imd_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                lable = os.path.basename(root).replace(" ", "-").lower()
                # print(lable,path)
                if not lable in lable_id:
                    lable_id[lable] = current_id
                    current_id = current_id + 1
                id_ = lable_id[lable]
                # x_train.append(path)
                # y_lable.append(lable)
                pil_img = Image.open(path).convert("L")  # grayscale
                size = (550, 550)
                final_img = pil_img.resize(size, Image.ANTIALIAS)
                img_array = np.array(pil_img, "uint8")
                # print(img_array)
                faces = face_haar_cascade.detectMultiScale(img_array, minNeighbors=5)

                for (x, y, w, h) in faces:
                    roi = img_array[y:y + h, x:x + w]
                    x_train.append(roi)
                    y_lable.append(id_)

    face_rec_id_pass={}        
    if os.path.isfile("labes.pickle"):
        pickle_in = open("labes.pickle", "rb")
        face_rec_id_pass = pickle.load(pickle_in)
        print(face_rec_id_pass)
        pickle_in.close()
    if len(face_rec_id_pass) == 0:
            face_rec_id_pass = lable_id
    else:
        f = lable_id
        face_rec_id_pass.update(f)

    pickle_out = open("labes.pickle", "wb")
    pickle.dump(face_rec_id_pass, pickle_out)

    pickle_out.close()
    print(face_rec_id_pass)        





    # print(x_train,y_lable)    
    reccognizer.train(x_train, np.array(y_lable))

    reccognizer.write("/Users/DELL/Desktop/faces recog/trainner.yml")
    reccognizer.save("/Users/DELL/Desktop/faces recog/trainner.yml")


