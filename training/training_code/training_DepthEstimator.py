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

from utils.hourglass_net_depth_singleStack import hourglass_refinement
from utils.IO import get_renderpeople_patch, get_camera, get_tiktok_patch, write_prediction, write_prediction_normal, save_prediction_png
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
    
    x1 = tf.placeholder(tf.float32, shape=(None, 256,256,9))
    y1 = tf.placeholder(tf.float32, shape=(None, 256,256,1))
    n1 = tf.placeholder(tf.float32, shape=(None, 256,256,3))
    z1 = tf.placeholder(tf.bool, shape=(None, 256,256,1))
    
    ## *****************************camera***********************************
    R = tf.placeholder(tf.float32, shape=(3,3))
    Rt = tf.placeholder(tf.float32, shape=(3,3))
    K = tf.placeholder(tf.float32, shape=(3,3))
    Ki = tf.placeholder(tf.float32, shape=(3,3))
    C = tf.placeholder(tf.float32, shape=(3,4))
    cen = tf.placeholder(tf.float32, shape=(3))
    origin = tf.placeholder(tf.float32, shape=(None, 2))
    scaling = tf.placeholder(tf.float32, shape=(None, 1))
    
    ## ****************************Network****************************
    with tf.variable_scope('hourglass_stack_fused_depth_prediction', reuse=tf.AUTO_REUSE):
        out2_1 = hourglass_refinement(x1,True)
        
    ## ****************************Loss RP****************************
    
    nmap1 = dmap_to_nmap(out2_1, Rt, R, Ki, cen, z1, origin, scaling)

    total_loss1_d = calc_loss(out2_1,y1,z1)

    total_loss2_d = calc_loss_d_refined_mask(out2_1,y1,z1)

    total_loss_n = calc_loss_normal2(nmap1,n1,z1)

    total_loss_rp = 2*total_loss1_d + total_loss2_d + total_loss_n
    
    ## ****************************Loss all****************************
    total_loss = total_loss_rp
    
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
ck_pnts_dir = "../training_progress/model/DepthEstimator"
log_dir = "../training_progress/"
Vis_dir_rp  = "../training_progress/visualization/DepthEstimator/Tang/"

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
    

    (_,loss_val,prediction1,nmap1_pred) = sess.run([train_step,total_loss,out2_1,nmap1],
                                                  feed_dict={x1:X_1,y1:Y1,n1:N1,z1:Z1,
                                                             Rt:Rt1n, Ki:Ki1n,cen:cen1n, R:R1n,
                                                             origin:origin1n,scaling:scaling1n})
    if itr%10 == 0:
        f_err = open(log_dir+"trainLog.txt","a")
        f_err.write("%d %g\n" % (itr,loss_val))
        f_err.close()
        print("")
        print("iteration %3d, depth refinement training loss is %g." %(itr,  loss_val))
        
    if itr % 100 == 0:
        fidx = [int(frms[0])]
        write_prediction(Vis_dir_rp,prediction1,itr,fidx,Z1);
        write_prediction_normal(Vis_dir_rp,nmap1_pred,itr,fidx,Z1)
        save_prediction_png (prediction1[0,...,0],nmap1_pred[0,...],X1,Z1,Z1_3,Vis_dir_rp,itr,fidx,0.99)

        
    if itr % 10000 == 0 and itr != 0:
        save_path = saver.save(sess,ck_pnts_dir+"/model_"+str(itr)+"/model_"+str(itr)+".ckpt")





