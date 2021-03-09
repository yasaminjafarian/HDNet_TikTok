"""
author: Yasamin Jafarian
"""

import tensorflow as tf
import os.path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import skimage.data
from PIL import Image, ImageDraw, ImageFont
import random
import sys
import matplotlib.pyplot as plt
tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import math
from tensorflow.python.platform import gfile
import scipy.misc

from hourglass_net_depth import hourglass_refinement
from hourglass_net_normal import hourglass_normal_prediction
from utils import (write_matrix_txt,get_origin_scaling,get_concat_h, depth2mesh, read_test_data, nmap_normalization, get_test_data) 

############################## test path and outpath ##################################
data_main_path = './test_data'
outpath = data_main_path+"/infer_out/"
visualization = True

##############################    Inference Code     ##################################
pre_ck_pnts_dir_DR =  "./model/depth_prediction"
model_num_DR = '1920000'
pre_ck_pnts_dir_NP =  "./model/normal_prediction"
model_num_NP = '1710000'
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

# Creat the outpath if not exists
Vis_dir = outpath
if not gfile.Exists(Vis_dir):
    print("Vis_dir created!")
    gfile.MakeDirs(Vis_dir)
refineNet_graph = tf.Graph()
NormNet_graph = tf.Graph()

# Define the depth and normal networks
# ***********************************Normal Prediction******************************************
with NormNet_graph.as_default():
    x1_n = tf.placeholder(tf.float32, shape=(None, 256,256,3))
    with tf.variable_scope('hourglass_normal_prediction', reuse=tf.AUTO_REUSE):
        out2_normal = hourglass_normal_prediction(x1_n,True)
# ***********************************Depth Prediction******************************************
with refineNet_graph.as_default():
    x1 = tf.placeholder(tf.float32, shape=(None, 256,256,9))
    with tf.variable_scope('hourglass_stack_fused_depth_prediction', reuse=tf.AUTO_REUSE):
        out2_1 = hourglass_refinement(x1,True)

# load checkpoints
sess2 = tf.Session(graph=NormNet_graph)
sess = tf.Session(graph=refineNet_graph)
with sess.as_default():
    with refineNet_graph.as_default():
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        saver = tf.train.import_meta_graph(pre_ck_pnts_dir_DR+'/model_'+model_num_DR+'/model_'+model_num_DR+'.ckpt.meta')
        saver.restore(sess,pre_ck_pnts_dir_DR+'/model_'+model_num_DR+'/model_'+model_num_DR+'.ckpt')
        print("Model DR restored.")
with sess2.as_default():
    with NormNet_graph.as_default():
        tf.global_variables_initializer().run()
        saver2 = tf.train.Saver()
        saver2 = tf.train.import_meta_graph(pre_ck_pnts_dir_NP+'/model_'+model_num_NP+'/model_'+model_num_NP+'.ckpt.meta')
        saver2.restore(sess2,pre_ck_pnts_dir_NP+'/model_'+model_num_NP+'/model_'+model_num_NP+'.ckpt')
        print("Model NP restored.")
        
# Read the test images and run the HDNet
test_files = get_test_data(data_main_path)

for f in range(len(test_files)):
    data_name = test_files[f]
    print('Processing file: ',data_name)
    X,Z, Z3, _, _,_,_, _,_,  _, _, DP = read_test_data(data_main_path,data_name,IMAGE_HEIGHT,IMAGE_WIDTH)
    
    prediction1n = sess2.run([out2_normal],feed_dict={x1_n:X})
    
    normal_pred_raw  = np.asarray(prediction1n)[0,...]
    normal_pred = nmap_normalization(normal_pred_raw)
    
    normal_pred = np.where(Z3,normal_pred,np.zeros_like(normal_pred))
    
    X_1 = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,9),dtype='f') 

    X_1[...,0]=X[...,0]
    X_1[...,1]=X[...,1]
    X_1[...,2]=X[...,2]
    X_1[...,3]=normal_pred[...,0]
    X_1[...,4]=normal_pred[...,1]
    X_1[...,5]=normal_pred[...,2]
    X_1[...,6]=DP[...,0]
    X_1[...,7]=DP[...,1]
    X_1[...,8]=DP[...,2]
    
    prediction1 = sess.run([out2_1],feed_dict={x1:X_1})
    image  = np.asarray(prediction1)[0,0,...,0]
    imagen = normal_pred[0,...]
        
    write_matrix_txt(image*Z[0,...,0],Vis_dir+data_name+".txt")
    write_matrix_txt(imagen[...,0]*Z[0,...,0],Vis_dir+data_name+"_normal_1.txt")
    write_matrix_txt(imagen[...,1]*Z[0,...,0],Vis_dir+data_name+"_normal_2.txt")
    write_matrix_txt(imagen[...,2]*Z[0,...,0],Vis_dir+data_name+"_normal_3.txt")
    depth2mesh(image*Z[0,...,0], Z[0,...,0], Vis_dir+data_name+"_mesh")
    if visualization:
        depth_map = image*Z[0,...,0]
        normal_map = imagen*Z3[0,...]
        min_depth = np.amin(depth_map[depth_map>0])
        max_depth = np.amax(depth_map[depth_map>0])
        depth_map[depth_map < min_depth] = min_depth
        depth_map[depth_map > max_depth] = max_depth
        
        normal_map_rgb = -1*normal_map
        normal_map_rgb[...,2] = -1*((normal_map[...,2]*2)+1)
        normal_map_rgb = np.reshape(normal_map_rgb, [256,256,3]);
        normal_map_rgb = (((normal_map_rgb + 1) / 2) * 255).astype(np.uint8);
        
        plt.imsave(Vis_dir+data_name+"_depth.png", depth_map, cmap="hot") 
        plt.imsave(Vis_dir+data_name+"_normal.png", normal_map_rgb) 
        
        d = np.array(scipy.misc.imread(Vis_dir+data_name+"_depth.png"),dtype='f')
        d = np.where(Z3[0,...]>0,d[...,0:3],255.0)
        n = np.array(scipy.misc.imread(Vis_dir+data_name+"_normal.png"),dtype='f')
        n = np.where(Z3[0,...]>0,n[...,0:3],255.0)
        final_im = get_concat_h(Image.fromarray(np.uint8(X[0,...])),Image.fromarray(np.uint8(d)))
        final_im = get_concat_h(final_im,Image.fromarray(np.uint8(n)))
        final_im.save(Vis_dir+data_name+"_results.png")
        
        os.remove(Vis_dir+data_name+"_depth.png")
        os.remove(Vis_dir+data_name+"_normal.png")
    
