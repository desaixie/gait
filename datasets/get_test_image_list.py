import glob
import os
#from py_utils import load_utils
import pickle


def getImagePath():
    user_root = '../CamControl3DScene'

    image_path = 'tmp'
    return os.path.join(user_root, image_path)

def get_test_list():
    user_root = '../CamControl3DScene'
    image_list_path = 'tmp/test.txt'
    image_path = 'tmp'

    
    #image_list = load_utils.load_string_list(os.path.join(user_root, image_list_path))
    image_list = [name for name in os.listdir(os.path.join(user_root, image_path)) ]
    image_path_list = []
    for s_image_name in image_list:
        s_full_image_path = os.path.join(user_root,image_path,s_image_name)
        if os.path.isfile(s_full_image_path):
            image_path_list.append(s_full_image_path)
    return image_path_list


def get_pdefined_anchors():
    user_root = '../CamControl3DScene'
    pdefined_anchor_file = 'datasets/pdefined_anchor.pkl'
    #print(os.path.join(user_root, pdefined_anchor_file))
    f = open(os.path.join(user_root, pdefined_anchor_file), 'rb')
    pdefined_anchors = pickle.load(f, encoding='bytes')
    f.close()
    return pdefined_anchors
