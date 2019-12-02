import numpy as np
import cv2
import os
import pickle


def img_to_vect(img_path, cluster_model):
    """
    将一张图片根据训练好的kmeans model转换为vect, 针对训练的图片。
    """

    img = cv2.imread(img_path)

    gray = to_gray(img)
    kp, desc = gen_surf_features(gray)

    clustered_desc = cluster_model.predict(desc)
    img_bow_hist = np.bincount(clustered_desc,
                               minlength=cluster_model.n_clusters)
    #转化为1*K的形式,K为字典的大小，即聚类的类别数
    return img_bow_hist.reshape(1,-1)


def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)
    return gray

def gen_surf_features(gray_img):
    surf = cv2.xfeatures2d.SURF_create(400)
    key_query, desc_query = surf.detectAndCompute(gray_img, None)
    return key_query, desc_query

def gen_all_surf_features(imgs):
    print('generating surf features...')
    img_descs = []
    index = 0
    for item in imgs:
        index+=1
        print('generating feature(SURF):'+str(index)+'/'+str(len(imgs)))
        img = cv2.imread(item)
        img = to_gray(img)
        key_query, desc_query = gen_surf_features(img)
        img_descs.append(desc_query)
    if not os.path.exists('data/img_descs.txt'):
        f=open('data/img_descs.txt','wb+')
        pickle.dump(img_descs, f, -1)

    return img_descs

#A despercated method.Used before using bof model.
def drawMatchesKnn_cv2(img1_gray,kp1,img2_gray,kp2,goodMatch):
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]

    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1:w1 + w2] = img2_gray

    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]

    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)

    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0,0,255))

    cv2.namedWindow("match",cv2.WINDOW_NORMAL)
