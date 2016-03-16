import cv2
import imutils
import numpy as np
import os
import csv
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.externals import joblib
from scipy.cluster.vq import *

np.set_printoptions(threshold=np.nan)
sift = cv2.xfeatures2d.SIFT_create()


def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]


def TestSampleFeaturesGeneratorWithLabel(train_path):
    stdSlr, k, voc = joblib.load("bof.pkl")
    training_names = mylistdir(train_path)
    image_paths = []
    image_classes = []
    for training_name in training_names:
        dir = os.path.join(train_path, training_name)
        class_path = imutils.imlist(dir)
        image_paths += class_path
        image_classes += [training_name] * len(class_path)
    des_list = []
    HH = []
    image_names = np.reshape(image_paths, (-1, 1))
    for image_path in image_paths:
        im = cv2.imread(image_path)
        if im == None:
            print "No such file {}\nCheck if the file exists".format(image_path)
            exit()
        kpts, des = sift.detectAndCompute(im, None)
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        kernel = np.ones((50, 50), np.float32) / 2500
        hsv = cv2.filter2D(hsv, -1, kernel)
        h_hue = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        H = []
        n_hue = sum(h_hue)
        for h in h_hue:
            hh = np.float32(float(h) / float(n_hue))
            H.append(hh)
        h_sat = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        temp = []
        temp.append(np.std(H, ddof=1))
        # H = []
        n_sat = sum(h_sat)
        for h in h_sat:
            hh = np.float32(float(h) / float(n_sat))
            H.append(hh)
        temp.append(np.std(H, ddof=1))
        HH.append(H)
        des_list.append((image_path, des))
    # Stack all the descriptors vertically in a numpy array
    # print des_list
    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[0:]:
        descriptors = np.vstack((descriptors, descriptor))
    #
    test_features = np.zeros((len(image_paths), k), "float32")
    for i in xrange(len(image_paths)):
        words, distance = vq(des_list[i][1], voc)
        for w in words:
            test_features[i][w] += 1

    # Scale the features
    test_features = stdSlr.transform(test_features)
    image_classes = np.reshape(image_classes, (-1, 1))
    test_features = np.append(test_features, HH, axis=1)
    res = np.append(test_features, image_classes, axis=1)
    res = np.append(image_names, res, axis=1)
    fl = open('TestFeatureWithLabel.csv', 'w')

    writer = csv.writer(fl)
    for values in res:
        writer.writerow(values)

    fl.close()
    return res


def getData(path):
    temp = TestSampleFeaturesGeneratorWithLabel(path)
    l = temp.shape[1] - 1
    feature = temp[:, 1:l]
    label = temp[:, l]
    address = temp[:, 0]
    return feature, label, address


def makePredction(testData):
    clf = joblib.load('model.pkl')
    prediction = clf.predict(testData)
    return prediction


def compareResult(predict, target, address):
    l = len(predict)
    i = 0
    match = 0.0
    mis = []
    while i < l:
        info = []
        if predict[i] == target[i]:
            match += 1
        else:
            info.append(address[i])
            info.append(predict[i])
            info.append(target[i])
            mis.append(info)
        i += 1
    score = float((100 * match) / (100 * l))
    # print predict
    print "The predition accuracy is:", score

    pillLabels = ["Atripla", "Cymbalta", "Epzicom", "Lexapro", "Prezista", "Tivicay", "Truvada", "Truvada_Cymbalta"]
    names = ["Atripla", "Cymbalta", "Epzicom", "Lexapro", "Prezista", "Tivicay", "Truvada", "Tru_Cym"]
    cm = confusion_matrix(target, predict, labels=pillLabels)

    recall = recall_score(target, predict, labels=pillLabels, average=None)
    precision = precision_score(target, predict, labels=pillLabels, average=None)

    print "Confusion matrix: "
    print "        ", " ".join("%-8s" % name for name in names), " Recall"
    ii = [0, 1, 2, 3, 4, 5, 6, 7]
    for index in ii:
        print "%-9s" % names[index], "      ".join("%3d" % value for value in cm[index]), "     %3.2f" % recall[index]
    print "Precision", "     ".join("%3.2f" % score for score in precision)
    return mis


def showMisImage(misList):
    rows = len(misList)
    i = 0
    while i < rows:
        img = cv2.imread(misList[i][0], 0)
        index = str(i+1)
        cv2.namedWindow(index + "Misclassified " + misList[i][1] + " to " + misList[i][2])
        cv2.imshow(index + "Misclassified " + misList[i][1] + " to " + misList[i][2], img)
        i += 1
    cv2.waitKey(0)


X, Y, A = getData("dataset/test")
Y = Y.ravel()
y = makePredction(X)
miss = compareResult(y, Y, A)
showMisImage(miss)
