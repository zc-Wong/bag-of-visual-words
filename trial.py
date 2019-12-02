import cv2
from utils import *
from DictionaryTrainer import *
from ImageRetrievor import *
import pickle

retrieve_url = 'E://UserData\car_images\A0RP77\A0RP77_20151129151034_6765527322.jpg'
database_url = 'E://UserData\car_images\A0RP77'

dictionary_trainer = DictionaryTrainer()
img_retriever = ImageRetrievor(database_url)
train_data = dictionary_trainer.train(img_retriever.archives)
print(dictionary_trainer.img_bow_hist.shape)
#
# vector = img_to_vect(retrieve_url, dictionary_trainer.cluster_model)
# print(vector)


# test_list = [np.array([1,2,3]), np.array([1,2,3])]
# f=open('test_list.txt','wb+')
# pickle.dump(test_list, f, -1)
# f=open('test_list.txt','rb+')
# read_list=pickle.load(f)
# f.close()
# print(read_list)

# img_descs = gen_all_surf_features([retrieve_url])
# print(img_descs[0].shape)


