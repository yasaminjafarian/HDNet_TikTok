## **********************  import **********************
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import os.path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from os import path
import numpy as np
import skimage.data
from PIL import Image, ImageDraw, ImageFont
import random
import scipy.misc

import math
from tensorflow.python.platform import gfile

from utils.hourglass_net_normal_singleStack import hourglass_normal_prediction
from utils.IO import get_renderpeople_patch, get_camera, get_tiktok_patch, write_prediction, write_prediction_normal, save_prediction_png_normal
from utils.Loss_functions import calc_loss_normal2, calc_loss, calc_loss_d_refined_mask
from utils.Geometry_MB import dmap_to_nmap
from utils.denspose_transform_functions import compute_dp_tr_3d_2d_loss2 

print("You are using tensorflow version ",tf.VERSION)
os.environ["CUDA_VISIBLE_DEVICES"]="0"

## ********************** change your variables **********************
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
BATCH_SIZE = 8
ITERATIONS = 100000000

rp_path = "../training_data/Tang_data/"
RP_image_range = range(0,188)
origin1n, scaling1n, C1n, cen1n, K1n, Ki1n, M1n, R1n, Rt1n = get_camera(BATCH_SIZE,IMAGE_HEIGHT)


## **************************** define the network ****************************
refineNet_graph = tf.Graph()
with refineNet_graph.as_default():
    
    ## ****************************RENDERPEOPLE****************************
    x1 = tf.placeholder(tf.float32, shape=(None, 256,256,3))
    n1 = tf.placeholder(tf.float32, shape=(None, 256,256,3))
    z1 = tf.placeholder(tf.bool, shape=(None, 256,256,1))

    with tf.variable_scope('hourglass_normal_prediction', reuse=tf.AUTO_REUSE):
        out2 = hourglass_normal_prediction(x1,True)
    total_loss_n = calc_loss_normal2(out2,n1,z1)
    total_loss = total_loss_n
    
    ## ****************************optimizer****************************
    train_step = tf.train.AdamOptimizer().minimize(total_loss)



##  ********************** initialize the network ********************** 
sess = tf.Session(graph=refineNet_graph)
with sess.as_default():
    with refineNet_graph.as_default():
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
#         saver = tf.train.import_meta_graph(pre_ck_pnts_dir+'/model_'+model_num+'/model_'+model_num+'.ckpt.meta')
#         saver.restore(sess,pre_ck_pnts_dir+'/model_'+model_num+'/model_'+model_num+'.ckpt')
        print("Model restored.")
        
        
##  ********************** make the output folders ********************** 
ck_pnts_dir = "../training_progress/model/NormalEstimator"
log_dir = "../training_progress/"
Vis_dir_rp  = "../training_progress/visualization/NormalEstimator/Tang/"

if not gfile.Exists(ck_pnts_dir):
    print("ck_pnts_dir created!")
    gfile.MakeDirs(ck_pnts_dir)
    
if not gfile.Exists(Vis_dir_rp):
    print("Vis_dir created!")
    gfile.MakeDirs(Vis_dir_rp)
    
if (path.exists(log_dir+"trainLog.txt")):
    os.remove(log_dir+"trainLog.txt")
    
##  ********************** Run the training **********************     
for itr in range(ITERATIONS):
    (X_1, X1, Y1, N1, 
     Z1, DP1, Z1_3,frms) = get_renderpeople_patch(rp_path, BATCH_SIZE, RP_image_range, 
                                                  IMAGE_HEIGHT,IMAGE_WIDTH)
    

    (_,loss_val,prediction1) = sess.run([train_step,total_loss,out2],
                                                  feed_dict={x1:X1,n1:N1,z1:Z1})
    if itr%10 == 0:
        f_err = open(log_dir+"trainLog.txt","a")
        f_err.write("%d %g\n" % (itr,loss_val))
        f_err.close()
        print("")
        print("iteration %3d, depth refinement training loss is %g." %(itr,  loss_val))
        
    if itr % 100 == 0:
        # visually compare the first sample in the batch between predicted and ground truth
        fidx = [int(frms[0])]
        write_prediction_normal(Vis_dir_rp,prediction1,itr,fidx,Z1)
        save_prediction_png_normal (prediction1[0,...],X1,Z1,Z1_3,Vis_dir_rp,itr,fidx)
        
    if itr % 10000 == 0 and itr != 0:
        save_path = saver.save(sess,ck_pnts_dir+"/model_"+str(itr)+"/model_"+str(itr)+".ckpt")



