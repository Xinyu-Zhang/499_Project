import cv2
import imutils
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from pylab import *
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.svm import NuSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
import os

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
    image_classes = np.reshape(image_classes, (-1, 1))
    im_features = np.append(im_features, HH, axis=1)
    res = np.append(im_features, image_classes, axis=1)
    # res = np.append(image_names, res, axis = 1)
    fl = open('FeatureSample.csv', 'w')

    writer = csv.writer(fl)
    for values in res:
        writer.writerow(values)

    fl.close()
    return im_features, image_classes


def compareClf(trainData, trainTarget):
    clf_RF = RandomForestClassifier(n_estimators=200)
    scores_RF = cross_validation.cross_val_score(clf_RF, trainData, trainTarget, cv=5)
    print ("Accuracy of RF: %0.2f " % scores_RF.mean())

    clf_SVM = NuSVC(nu=0.1, kernel='rbf', probability=True)
    scores_SVM = cross_validation.cross_val_score(clf_SVM, trainData, trainTarget, cv=5)
    print ("Accuracy of SVM: %0.2f " % scores_SVM.mean())

    clf_NB = BernoulliNB(alpha=0.1)
    scores_NB = cross_validation.cross_val_score(clf_NB, trainData, trainTarget, cv=5)
    print ("Accuracy of NB: %0.2f " % scores_NB.mean())

    clf_AB = AdaBoostClassifier(base_estimator=clf_NB, algorithm="SAMME", n_estimators=200)
    scores_AB = cross_validation.cross_val_score(clf_NB, trainData, trainTarget, cv=5)
    print ("Accuracy of AB: %0.2f " % scores_AB.mean())

    clfs = [clf_RF, clf_SVM, clf_NB, clf_AB]
    scores = [scores_RF.mean(), scores_SVM.mean(), scores_NB.mean(), scores_AB.mean()]
    index = argmax(scores)
    print 'After comparasion, the best is', clfs[index]
    return clfs[index]


def catchData():
    trainData, trainTarget = TrainingSampleFeaturesGenerator("dataset/train")
    trainTarget = trainTarget.ravel()
    return trainData, trainTarget


def saveModel(clfChosen, trainData, trainTarget):
    clfSave = clfChosen.fit(trainData, trainTarget)
    joblib.dump(clfSave, 'model.pkl', compress=3)
    print "Model saved in 'model.pkl'. "


X, Y = catchData()
model = compareClf(X, Y)
saveModel(model, X, Y)
