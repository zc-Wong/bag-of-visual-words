from ImageRetrievor import *
import cv2
from utils import *
import os
# from scipy.interpolate import spline
import matplotlib.pyplot as plt

database_url = 'C://Users\Administrator\Desktop\ImageRetrievor\data\image'
#retrieve_url = 'C://Users\Aministrator\Desktop\ImageRetrievor\data\image\\bark\img1.jpg'
retrieve_url = 'D:\\bark\\bark1.jpg'
#retrieve_url = 'E://UserData\car_imagest\A1TA21\A1TA21_20151201195217_3091692917.jpg'


def drawRecallPrecisonCurve(imageRetrievor, maxN):
    power_precison = []
    power_recall = []

    for index in range(maxN):
        power_precison.append(imageRetrievor.computePrecisonRate(index+1))
        power_recall.append(imageRetrievor.computeRecallRate(index+1))
    print("average precision is", np.average(power_precison))
    print("final recall rate is", power_recall[-1])
    power_precison = np.array(power_precison)
    power_recall = np.array(power_recall)

#     T = np.array(range(maxN))
#     xnew = np.linspace(T.min(),T.max(),300)
#     power_smooth_precison = spline(T,power_precison,xnew)
#     power_smooth_call = spline(T,power_recall,xnew)
#
#     plt.plot(xnew,power_smooth_precison)
#     plt.plot(xnew,power_smooth_call)
#     plt.show()

maxN = 10
imageRetrievor = ImageRetrievor(database_url, maxN)
imageRetrievor.retrieve(retrieve_url)
drawRecallPrecisonCurve(imageRetrievor, imageRetrievor.maxN)
imageRetrievor.query_expansion()
drawRecallPrecisonCurve(imageRetrievor, imageRetrievor.maxN)

#imageRetrievor.drawRecallRate(return_N=5)
#imageRetrievor.drawPrecisonRate(return_N=5)
# for mindistance in mindistances:
#     print(imageRetrievor.archives[mindistance[0]])
#     img = cv2.imread(imageRetrievor.archives[mindistance[0]])


#for index in range(maxN):
    #print("original detected:"+imageRetrievor.archives[imageRetrievor.min_distances[index][0]]+" K="+str(imageRetrievor.img_class[index]))


