import argparse as ap
import cv2
import imutils
import numpy as np
import os
import csv
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

np.set_printoptions(threshold=np.nan)
sift = cv2.xfeatures2d.SIFT_create()


def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]


# given training sample path, return extracted features and corresponding feature label
def TrainingSampleFeaturesGenerator(train_path):
    training_names = mylistdir(train_path)
    image_paths = []
    image_classes = []
    class_id = 0
    for training_name in training_names:
        dir = os.path.join(train_path, training_name)
        class_path = imutils.imlist(dir)
        image_paths += class_path
        image_classes += [training_name] * len(class_path)

    # List where all the descriptors are stored
    des_list = []
    HH = []
    for image_path in image_paths:
        im = cv2.imread(image_path)
        kpts, des = sift.detectAndCompute(im, None)
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        kernel = np.ones((50, 50), np.float32) / 2500
        hsv = cv2.filter2D(hsv, -1, kernel)
        H = []
        h_hue = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        n_hue = sum(h_hue)
        for h in h_hue:
            hh = np.float32(float(h) / float(n_hue))
            H.append(hh)

        h_sat = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        n_sat = sum(h_sat)
        for h in h_sat:
            hh = np.float32(float(h) / float(n_sat))
            H.append(hh)
        HH.append(H)
        des_list.append((image_path, des))

    print HH
    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    # Perform k-means clustering
    k = 80
    voc, variance = kmeans(descriptors, k, 1)

    # Calculate the histogram of features
    im_features = np.zeros((len(image_paths), k), "float32")
    for i in xrange(len(image_paths)):
        words, distance = vq(des_list[i][1], voc)
        for w in words:
            im_features[i][w] += 1
    # Scaling the words
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)

    # Save the SVM
    joblib.dump((stdSlr, k, voc), "bof.pkl", compress=3)
    print len(im_features)
    print len(image_classes)
    image_classes = np.reshape(image_classes, (-1, 1))
    im_features = np.append(im_features, HH, axis=1)
    res = np.append(im_features, image_classes, axis=1)
    fl = open('FeatureSample.csv', 'w')

    writer = csv.writer(fl)
    for values in res:
        writer.writerow(values)

    fl.close()
    return im_features, image_classes


# given test sample, return test sample features,
def TestSampleFeaturesGenerator(image_path):
    stdSlr, k, voc = joblib.load("bof.pkl")

    image_paths = imutils.imlist(image_path)
    # List where all the descriptors are stored
    des_list = []
    HH = []
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
        n_sat = sum(h_sat)
        for h in h_sat:
            hh = np.float32(float(h) / float(n_sat))
            H.append(hh)
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
    test_features = np.append(test_features, HH, axis=1)
    fl = open('TestFeature.csv', 'w')

    writer = csv.writer(fl)
    for values in test_features:
        writer.writerow(values)

    fl.close()
    return test_features


# im_features, image_classes = TrainingSampleFeaturesGenerator("dataset/train")
# test_features = TestSampleFeaturesGenerator("dataset/test")


# g1 = np.uint8([[[150, 120, 150]]])
# hsv_g1 = cv2.cvtColor(g1, cv2.COLOR_BGR2HSV)
# print hsv_g1
#
# g2 = np.uint8([[[195, 190, 200]]])
# hsv_g2 = cv2.cvtColor(g2, cv2.COLOR_BGR2HSV)
# print hsv_g2
#
# filepath = "data/test/Cymbalta_10.jpg"
# # hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
# img = cv2.imread('Lexapro_01.jpg')
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# kernel = np.ones((50, 50), np.float32) / 2500
# dst = cv2.filter2D(hsv, -1, kernel)
#
# h_hue = cv2.calcHist([dst], [0], None, [180], [0, 180])
# H = []
# n_hue = sum(h_hue)
# for h in h_hue:
#     hh = np.float32(float(h) / float(n_hue))
#     H.append(hh)
# h_sat = cv2.calcHist([dst], [1], None, [256], [0, 256])
# n_sat = sum(h_sat)
# for h in h_sat:
#     hh = np.float32(float(h) / float(n_sat))
#     H.append(hh)
#
# X = []
# i = 0
# for a in H:
#     X.append(i)
#     i = i + 1
#
# plt.plot(X, H)
# plt.show()
#

# path = "dataset/test/"
# #GBR
# image_paths = mylistdir(path)
# print image_paths
# for filename in image_paths:
# 	print filename
# 	image = cv2.imread(path + filename)
# 	hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

# 	lower_blue = np.array([20,0,80])
# 	upper_blue = np.array([180,255,255])

# 	 # Threshold the HSV image to get only blue colors
# 	mask = cv2.inRange(hsv, lower_blue, upper_blue)

# 	# Bitwise-AND mask and original image
# 	res = cv2.bitwise_and(image,image, mask= mask)
# 	cv2.imshow('res',res)
# 	cv2.waitKey(0)
