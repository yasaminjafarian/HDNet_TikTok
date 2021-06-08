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
BATCH_SIZE = 1
ITERATIONS = 100000000

pre_ck_pnts_dir = "../model/depth_prediction"
model_num = '1920000'
model_num_int = 1920000

rp_path = "../training_data/Tang_data/"
tk_path = "../training_data/tiktok_data/"
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
    
    ## ****************************tiktok****************************
    x1_tk = tf.placeholder(tf.float32, shape=(None, 256,256,9))
    n1_tk = tf.placeholder(tf.float32, shape=(None, 256,256,3))
    z1_tk = tf.placeholder(tf.bool, shape=(None, 256,256,1))
    
    x2_tk = tf.placeholder(tf.float32, shape=(None, 256,256,9))
    n2_tk = tf.placeholder(tf.float32, shape=(None, 256,256,3))
    z2_tk = tf.placeholder(tf.bool, shape=(None, 256,256,1))
    
    i_r1_c1_r2_c2 = tf.placeholder(tf.int32, shape=(None, 25000,5))
    i_limit = tf.placeholder(tf.int32, shape=(None, 24,3))
    
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
        
    with tf.variable_scope('hourglass_stack_fused_depth_prediction', reuse=tf.AUTO_REUSE):
        out2_1_tk = hourglass_refinement(x1_tk,True)
        
    with tf.variable_scope('hourglass_stack_fused_depth_prediction', reuse=tf.AUTO_REUSE):
        out2_2_tk = hourglass_refinement(x2_tk,True)

        
    ## ****************************Loss RP****************************
    
    nmap1 = dmap_to_nmap(out2_1, Rt, R, Ki, cen, z1, origin, scaling)

    total_loss1_d = calc_loss(out2_1,y1,z1)

    total_loss2_d = calc_loss_d_refined_mask(out2_1,y1,z1)

    total_loss_n = calc_loss_normal2(nmap1,n1,z1)

    total_loss_rp = 2*total_loss1_d + total_loss2_d + total_loss_n
    
    ## ****************************Loss TK****************************
    
    nmap1_tk = dmap_to_nmap(out2_1_tk, Rt, R, Ki, cen, z1_tk, origin, scaling)
    nmap2_tk = dmap_to_nmap(out2_2_tk, Rt, R, Ki, cen, z2_tk, origin, scaling)

    total_loss_n_tk = calc_loss_normal2(nmap1_tk,n1_tk,z1_tk)+calc_loss_normal2(nmap2_tk,n2_tk,z2_tk)
    
    loss3d,loss2d,PC2p,PC1_2 = compute_dp_tr_3d_2d_loss2(out2_1_tk,out2_2_tk,
                                                         i_r1_c1_r2_c2[0,...],i_limit[0,...],
                                                         C,R,Rt,cen,K,Ki,origin,scaling)

    total_loss_tk = total_loss_n_tk + 5*loss3d

    ## ****************************Loss all****************************
    total_loss = total_loss_rp+total_loss_tk
    
    ## ****************************optimizer****************************
    train_step = tf.train.AdamOptimizer(learning_rate=0.001,
                                        beta1=0.9,
                                        beta2=0.999,
                                        epsilon=0.1,
                                        use_locking=False,
                                        name='Adam').minimize(total_loss)

##  ********************** initialize the network ********************** 
sess = tf.Session(graph=refineNet_graph)
with sess.as_default():
    with refineNet_graph.as_default():
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        saver = tf.train.import_meta_graph(pre_ck_pnts_dir+'/model_'+model_num+'/model_'+model_num+'.ckpt.meta')
        saver.restore(sess,pre_ck_pnts_dir+'/model_'+model_num+'/model_'+model_num+'.ckpt')
        print("Model restored.")
        
##  ********************** make the output folders ********************** 
ck_pnts_dir = "../training_progress/model/HDNet"
Vis_dir  = "../training_progress/visualization/HDNet/tiktok/"
log_dir = "../training_progress/"
Vis_dir_rp  = "../training_progress/visualization/HDNet/Tang/"

if not gfile.Exists(ck_pnts_dir):
    print("ck_pnts_dir created!")
    gfile.MakeDirs(ck_pnts_dir)

if not gfile.Exists(Vis_dir):
    print("Vis_dir created!")
    gfile.MakeDirs(Vis_dir)
    
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
    (X_1_tk, X1_tk, N1_tk, Z1_tk, DP1_tk, Z1_3_tk, 
     X_2_tk, X2_tk, N2_tk, Z2_tk, DP2_tk, Z2_3_tk, 
     i_r1_c1_r2_c2_tk, i_limit_tk, 
     frms_tk, frms_neighbor_tk) = get_tiktok_patch(tk_path, BATCH_SIZE, IMAGE_HEIGHT,IMAGE_WIDTH)


    (_,loss_val,prediction1,nmap1_pred, 
     prediction1_tk,nmap1_pred_tk,PC2pn,PC1_2n) = sess.run([train_step,total_loss,out2_1,
                                                           nmap1,out2_1_tk,nmap1_tk,PC2p,PC1_2],
                                                  feed_dict={x1:X_1,y1:Y1,n1:N1,z1:Z1,
                                                             Rt:Rt1n, Ki:Ki1n,cen:cen1n, R:R1n,
                                                             origin:origin1n,scaling:scaling1n,
                                                             x1_tk:X_1_tk,n1_tk:N1_tk,z1_tk:Z1_tk,
                                                             x2_tk:X_2_tk,n2_tk:N2_tk,z2_tk:Z2_tk,
                                                             i_r1_c1_r2_c2:i_r1_c1_r2_c2_tk,
                                                             i_limit:i_limit_tk})
    if itr%10 == 0:
        f_err = open(log_dir+"trainLog.txt","a")
        f_err.write("%d %g\n" % (itr,loss_val))
        f_err.close()
        print("")
        print("iteration %3d, depth refinement training loss is %g." %(itr,  loss_val))
        
    if itr % 100 == 0:
        # visually compare the first sample in the batch between predicted and ground truth
        fidx = [int(frms[0])]
        write_prediction(Vis_dir_rp,prediction1,itr,fidx,Z1);
        write_prediction_normal(Vis_dir_rp,nmap1_pred,itr,fidx,Z1)
        save_prediction_png (prediction1[0,...,0],nmap1_pred[0,...],X1,Z1,Z1_3,Vis_dir_rp,itr,fidx,1)
        fidx = [int(frms_tk[0])]
        write_prediction(Vis_dir,prediction1_tk,itr,fidx,Z1_tk);
        write_prediction_normal(Vis_dir,nmap1_pred_tk,itr,fidx,Z1_tk)
        save_prediction_png (prediction1_tk[0,...,0],nmap1_pred_tk[0,...],X1_tk,Z1_tk,Z1_3_tk,Vis_dir,itr,fidx,1)

    if itr % 10000 == 0 and itr != 0:
        save_path = saver.save(sess,ck_pnts_dir+"/model_"+str(itr)+"/model_"+str(itr)+".ckpt")






