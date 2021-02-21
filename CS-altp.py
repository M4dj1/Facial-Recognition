import numpy as np
import pylab as plt
import matplotlib.image as mpimg
import sys
import re
from PIL import Image
import os
from scipy.spatial import distance

trainSet = []

def sa(value, t):
    if value >= t:
        return 1
    elif abs(value) < t:
        return 0
    elif value <= -t:
        return -1

def padding(a):
    zeroH = np.zeros(a.shape[1] + 2).reshape(1, a.shape[1] + 2)
    zeroV = np.zeros(a.shape[0]).reshape(a.shape[0], 1)
    a = np.concatenate((a, zeroV), axis=1)
    a = np.concatenate((zeroV, a), axis=1)
    a = np.concatenate((zeroH, a), axis=0)
    a = np.concatenate((a, zeroH), axis=0)
    return a

def split(array, nrows, ncols):
    h, w = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))

def csaltp_training(image, k, test):
    T = np.floor(np.multiply(image, k))
    image = padding(image)
    T = padding(T)
    resultImgNeg = np.zeros((image.shape[0], image.shape[1]))
    resultImgPos = np.zeros((image.shape[0], image.shape[1]))
    ss = np.zeros(4).astype(int)
    for x in range(2, image.shape[0] - 1):
        for y in range(2, image.shape[1] - 1):
            ss[0] = sa(image[x - 1][y - 1] - image[x+1][y+1], T[x+1][y+1])
            ss[1] = sa(image[x - 1][y] - image[x+1][y], T[x+1][y])
            ss[2] = sa(image[x - 1][y + 1] - image[x+1][y-1], T[x+1][y-1])
            ss[3] = sa(image[x][y + 1] - image[x][y-1], T[x][y-1])
            sneg = ss.copy()
            spos = ss.copy()
            for i in range(ss.shape[0]):
                if (ss[i] == 1):
                    sneg[i] = 0
                elif (ss[i] == -1):
                    sneg[i] = 1
                    spos[i] = 0
            resultImgNeg[x][y] = sneg[0] + sneg[1]*2 + sneg[2]*4 + sneg[3]*8
            resultImgPos[x][y] = spos[0] + spos[1]*2 + spos[2]*4 + spos[3]*8
    resultImgNeg = resultImgNeg[1:image.shape[0] - 1, 1:image.shape[1] - 1].astype(int)
    resultImgPos = resultImgPos[1:image.shape[0] - 1, 1:image.shape[1] - 1].astype(int)
    image4x4Neg = split(resultImgNeg,23,28)
    image4x4Pos = split(resultImgPos,23,28)
    histOfHistNeg = []
    histOfHistPos = []
    for idx ,cell in enumerate(image4x4Neg):
        cell.flatten()
        hist = np.zeros(16).astype(int)
        for i in cell:
            hist[i] = hist[i] + 1
        histOfHistNeg.append(hist)
    for idx ,cell in enumerate(image4x4Pos):
        hist = np.zeros(16).astype(int)
        for i in cell:
            hist[i] = hist[i] + 1
        histOfHistPos.append(hist)
    finalfeature = np.array([np.array(histOfHistNeg),np.array(histOfHistPos)])
    if test == 'true':
        trainSet.append(finalfeature)
    return finalfeature

def csaltp_testing(filename, k):
    imgOri = Image.open('Faces/TestSet/' + filename).convert('L')
    imgMat = np.array(imgOri)
    imgOutHistogram = csaltp_training(imgMat, k, test='false')
    return imgOutHistogram

def calcDistance(trainset, test):
    distances = [float('inf')] * 40
    for x in range(trainset.shape[0]):
        dist = distance.euclidean(trainset[x].flatten(), test.flatten())
        distances.append(dist)
    return np.array(distances)

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

def console():
    def show():
        def process(filename: str = None) -> None:
            image = Image.open(filename)
            ax[j][i].set_title("Knn = "+str(valueKNN[j-1]), color='red')
            ax[j][i].imshow(image, cmap='gray')
        fig, ax = plt.subplots(nrows=6, ncols=10)
        for x in range(6):
            for y in range(10):
                ax[x][y].set_axis_off()
        j , i = [0 , 0]
        image = Image.open('Faces/TestSet/' + str(TestPicture) + '.jpg')
        ax[j][i].set_title("Test Image", color='Green')
        ax[j][i].set_axis_off()
        ax[j][i].imshow(image, cmap='gray')
        j += 1
        for K in Knn:
            for file in K:
                process('Faces/TrainSet/' + str(file+1) + '.jpg')
                i += 1
            j = j + 1
            i = 0
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()
    weberK = float(input("Enter Weber's K value : "))
    print()

    dir = sorted_alphanumeric(os.listdir('Faces/TrainSet/'))
    for idx, filename in enumerate(dir):
        count = int((idx/360)*100)
        sys.stdout.write("Training : [" + str(count)+"%]")
        sys.stdout.flush()
        imgOri = Image.open('Faces/TrainSet/' + filename).convert('L')
        imgMat = np.array(imgOri)
        csaltp_training(imgMat, weberK, test='true')
        sys.stdout.write("\r")
    sys.stdout.write("Training Done!!\n") # this ends the progress bar
    print()

    TestPicture = input("Enter Test picture index (1 -> 40) or '0' to exit : ")
    while TestPicture != '0' :
        imgOutHist = csaltp_testing(TestPicture + '.jpg', weberK)
        Knn = []
        distTrainTest = calcDistance(np.array(trainSet), imgOutHist)
        valueKNN = np.array([1 ,3 ,5, 7, 10])
        for i in range(valueKNN.shape[0]):
            Knn.append(np.argpartition(distTrainTest, range(valueKNN[i]))[:valueKNN[i]])
        show()
        TestPicture = input("Enter Test picture index (1 -> 40) or '0' to exit : ")

console()
