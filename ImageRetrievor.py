# -*- coding: utf-8 -*-

from PIL import Image
from skimage import transform,data
import numpy as np
from scipy.spatial.distance import pdist
import cv2
import os
import math
from DictionaryTrainer import *

class ImageRetrievor(object):
    def __init__(self, database_url, maxN):
        self.archives = []
        self.distances = []

        self.database_url = database_url
        self.dictionary_trainer = DictionaryTrainer(K=256)

        self.std_width = 136 * 3
        self.std_height = 76 * 3

        self.retrieve_vectors = []

        self.min_distances = []
        self.img_class = []
        self.maxN = maxN
        self.average_N = 10
        # maxN

        self.type='BoW'

    def retrieve(self, imageUrl):
        self.image_input(imageUrl, self.type)
        self.compute_archives()
        self.compute_retrieve_vectors(self.type)
        self.compute_distance(self.type)
        self.min_distances = sorted(enumerate(self.distances), key=lambda x:x[1])
        self.resort_process()

        minIndex = self.distances.index(min(self.distances))
        file_name = self.archives[minIndex]
        img = cv2.imread(file_name)
        cv2.imshow("best_result",img)
        cv2.waitKey(0)

    def query_expansion(self):
        print("Start Query expansion--------------------------------------------")
        new_input_hist = np.zeros(self.dictionary_trainer.K)
        for index in range(self.average_N):
            new_input_hist += self.retrieve_vectors[self.min_distances[index][0]]
        new_input_hist /= self.average_N#len(self.retrieve_vectors)

        self.vector = new_input_hist
        self.compute_distance()
        #print(self.distances)
        self.min_distances = sorted(enumerate(self.distances), key=lambda x:x[1])
        #print(len(self.min_distances))
        print("Finish Query expansion--------------------------------------------")


    def image_input(self, imageUrl, type):
        self.image_url = imageUrl
        if (type=='flatten'):
            self.image = Image.open(imageUrl)
            self.image_array = np.array(self.image)
            self.vector = transform.resize(self.image_array, (self.std_width, self.std_height)).flatten()
        elif (type=='BoW'):
            self.vector = None #Can only be computed after training process done.

    def compute_distance(self, type='BoW', p=3):
        print('computing distance...')
        index = 0
        self.distances = []
        for retrieve_vector in self.retrieve_vectors:
            index+=1

            if (type=='eu' or type=='BoW'):
                self.distances.append(np.linalg.norm(self.vector-retrieve_vector))
            elif (type=='min'):
                self.distances.append(pdist(np.vstack([self.vector, retrieve_vector]),'minkowski',p)[0])
            elif (type=='cos'):
                num = float(self.vector.dot(retrieve_vector))
                denom = np.linalg.norm(self.vector) * np.linalg.norm(retrieve_vector)
                cos = num / denom #余弦值
                sim = 0.5 + 0.5 * cos #归一化
                self.distances.append(sim)
            elif (type=='SIFT'):
                realDis = .0
                for m in retrieve_vector:
                    realDis+=m[0].distance
                realDis /= len(retrieve_vector)
                self.distances.append(realDis)

    def compute_retrieve_vectors(self, type='BoW'):
        #最原始的方案 直接展开图片作为特征向量
        if type=='Original':
            for retrieve_img_url in self.archives:
                retrieve_img = Image.open(retrieve_img_url)
                retrieve_vector = transform.resize(np.array(retrieve_img), self.image_array.shape).flatten()
                self.retrieve_vectors.append(retrieve_vector)
        #将图片与数据库中图片的SIFT特征匹配作为特征向量（两张图片算距离时求它们匹配之间的平均距离）
        if type=='SIFT':
            index = 0
            for retrieve_img_url in self.archives:
                index+=1
                print('computing retrievector(SIFT):'+str(index)+'/'+str(len(self.archives)))
                kp_ret, des_ret = self.compute_features(retrieve_img_url)
                kp_or, des_or = self.compute_features(self.image_url)
                bf = cv2.BFMatcher(cv2.NORM_L2)
                matches = bf.knnMatch(des_ret, des_or, k = 1)
                self.retrieve_vectors.append(matches)
        #把特征进一步转换为直方图（Use BoW）作为特征向量 也是我们最终使用的方法.
        if type=='BoW':
            #Get pure surf features.
            if os.path.exists('data/img_descs.txt'):
                print('there exists descs.Use it.')
                f=open('data/img_descs.txt','rb+')
                self.img_descs=pickle.load(f)
                f.close()
            else:
                print('Missing hist or missing img_descs(features).Compute the hist using database provided.')
                self.img_descs = gen_all_surf_features(self.archives)


            self.dictionary_trainer.train(self.img_descs)
            self.vector = img_to_vect(self.image_url, self.dictionary_trainer.cluster_model)[0].astype(np.float64)#Until now can we compute the vector of origin retrieve image
            self.img_bow_hist = self.dictionary_trainer.img_bow_hist.astype(np.float64)
            self.TF_IDF_modify()    #vector&hist exists difference between trainer(origin) && retrievor(improved)
            for index in range(len(self.archives)):
                self.retrieve_vectors.append(self.img_bow_hist[index])#Use improved vector&hist.

    def resort_process(self):
        resort_img_descs = []
        trainer = DictionaryTrainer(32)
        for index in range(self.maxN):
            resort_img_descs.append(self.img_descs[self.min_distances[index][0]])

        trainer.train(resort_img_descs)
        resort_img_bow_hists = trainer.img_bow_hist
        for resort_img_bow_hist in resort_img_bow_hists:
            resort_img_bow_hist_list = resort_img_bow_hist.tolist()
            self.img_class.append(resort_img_bow_hist_list.index(max(resort_img_bow_hist_list)))

        #Use this hist to judge which class is better.
        retrieve_hist = img_to_vect(self.image_url, trainer.cluster_model)[0].tolist()
        #从大到小排序 每一项（类别，对应类别的值（置信值））
        retrieve_max_hist = sorted(enumerate(retrieve_hist), key=lambda x:x[1], reverse=True)
        #print(retrieve_max_hist)
        for k in retrieve_max_hist:
            for index in range(len(self.img_class)):
                if k[0]==self.img_class[index]:
                    #pass
                    print("detected:"+self.archives[self.min_distances[index][0]]+" K="+str(self.img_class[index]))

    def TF_IDF_modify(self):
        print('TF_IDF modifing...')
        h_sum = np.sum(self.vector)
        h_hist = [hj/h_sum for hj in self.vector]
        N = len(self.archives)
        for index in range(len(self.vector)):
            fj = 0.1    #divide zero
            for img_hist in self.img_bow_hist:
                if self.vector[index]==img_hist[index]:
                    fj+=1
            #print("to-"+str(h_hist[index] * math.log(N/fj)))
            self.vector[index] = h_hist[index] * math.log(N/fj)

        for index in range(len(self.img_bow_hist)):
            hist_sum = np.sum(self.img_bow_hist[index])
            for j in range(len(self.img_bow_hist[index])):
                self.img_bow_hist[index][j] /= hist_sum

        #print(type(self.vector))
        #print(np.sum(self.vector))

    #A despercated method.Used before using bof model.
    def compute_features(self, imgUrl, type='SIFT'):
        img = cv2.imread(imgUrl)
        img = cv2.resize(img,(136 * 3,76 * 3))
        # cv2.imshow("original",img)
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

        if type=='SIFT':
            #使用SIFT
            sift = cv2.xfeatures2d.SIFT_create()
            self.keypoints, descriptor = sift.detectAndCompute(gray,None)
            # cv2.drawKeypoints(image = img,
            #                   outImage = img,
            #                   keypoints = self.keypoints,
            #                   flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            #                   color = (51,163,236))
            # cv2.imshow("SIFT",img)

        elif type=='SURF':
            surf = cv2.xfeatures2d.SURF_create()
            self.keypoints, descriptor = surf.detectAndCompute(gray,None)
            # cv2.drawKeypoints(image = img,
            #                   outImage = img,
            #                   keypoints = self.keypoints,
            #                   flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            #                   color = (51,163,236))
            # cv2.imshow("SURF",img)

        return self.keypoints, descriptor

    def compute_archives(self):
        print('computing archives...')
        for root ,dirs ,files in os.walk(self.database_url):
            for file in files:
                file_name = os.path.join(root,file)
                if file_name.split("\\")[-1].split('.')[-1] =='jpg' and file_name!=self.image_url:
                    self.archives.append(file_name)
        print('computing archives done')

    def computeRecallRate(self, return_N=10):
        className = self.image_url.split('\\')[-2]
        realName = self.image_url.split('\\')[-1]
        computeCorrectNum = 0

        root_Url = self.database_url+"\\"+className
        #If exists no correct picture. Divide zero make it exception.
        actualCorrectNum = len([name for name in os.listdir(root_Url) if os.path.isfile(os.path.join(root_Url, name))])
        for index in range(return_N):
            #print("dected:"+self.archives[self.min_distances[index][0]])
            if self.archives[self.min_distances[index][0]].split('\\')[-2]==className:
                computeCorrectNum = computeCorrectNum+1

        print('compute correct num:'+str(computeCorrectNum)+"/actual correct num"+str(actualCorrectNum))
        print("RECALL RATE:"+str(computeCorrectNum/actualCorrectNum))
        return computeCorrectNum/actualCorrectNum

    def computePrecisonRate(self, return_N=10):
        className = self.image_url.split('\\')[-2]
        realName = self.image_url.split('\\')[-1]
        computeCorrectNum = 0

        root_Url = self.database_url+"\\"+className
        for index in range(return_N):
            #print("dected:"+self.archives[self.min_distances[index][0]])
            if self.archives[self.min_distances[index][0]].split('\\')[-2]==className:
                computeCorrectNum = computeCorrectNum+1

        print('compute correct num:'+str(computeCorrectNum)+"/return num"+str(return_N))
        print("PRECISION RATE:"+str(computeCorrectNum/return_N))
        return computeCorrectNum/return_N
