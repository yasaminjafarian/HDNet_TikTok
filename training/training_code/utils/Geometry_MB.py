import tensorflow as tf
import numpy as np
import skimage.data
from PIL import Image, ImageDraw, ImageFont
import math
from tensorflow.python.platform import gfile
import scipy.misc
from utils.vector import cross

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

# *****************************************************************************************************
def depth2points3D_MB(output, Rt, Ki, cen, z_r, origin, scaling):
    
    # Rt and Ki are 3*3, cen is 1*3, z_r and output are B*256*256, origin is B*2, scaling is B*1
    # point3D is 3*N
    
    ones_mat = tf.cast(tf.ones_like(z_r), tf.bool)
    indices = tf.where(ones_mat) #  (256*256*B) x 3 = N x 3
#     indices = tf.where(z_r)
    
    good_output = tf.where(z_r, output, 1000000*tf.ones_like(output))
    Dlambda = tf.gather_nd(good_output,indices) # N x 3

    num_of_points = tf.shape(Dlambda)[0] #N
    num_of_batches = tf.shape(z_r)[0]
    num_of_points_in_each_batch = tf.cast(tf.divide(num_of_points,num_of_batches),tf.int32)
    
    
    Dlambda_t = tf.reshape(Dlambda,[1,num_of_points]) # 1 x N
    Dlambda3 = tf.concat([Dlambda_t,Dlambda_t],0)
    Dlambda3 = tf.concat([Dlambda3,Dlambda_t],0) # 3 x N
        
    
    idx = tf.cast(indices, tf.float32)

    row_of_ones = tf.ones([1, num_of_points], tf.float32) # 1 x N
    
# dividing xy and the batch number    
    bxy = idx # N x 3
    b = tf.transpose(bxy[...,0]) # 1 x N
    batches = tf.reshape(b,[1,num_of_points]) # 1 x N
    xy = tf.transpose(tf.reverse(bxy[...,1:3],[1])) # 2 x N

# tiling the scaling to match the data
    scaling2 = tf.reshape(scaling, [num_of_batches,1])
    tiled_scaling = tf.tile(scaling2, [tf.constant(1),num_of_points_in_each_batch])
    scaling_row = tf.reshape(tiled_scaling,[1,num_of_points])   
    scaling_2_rows = tf.concat([scaling_row,scaling_row],0)

# scaling the input 
    scaled_xy = tf.multiply(xy, scaling_2_rows)

# dividing the origin 0 and origin 1 of the origin 
    origin0 = origin[...,0]
    origin0 = tf.reshape(origin0,[num_of_batches,1])
    origin1 = origin[...,1]
    origin1 = tf.reshape(origin1,[num_of_batches,1])
    
# tiling the origin0 to match the data
    tiled_origin0= tf.tile(origin0, [tf.constant(1),num_of_points_in_each_batch])
    origin0_row = tf.reshape(tiled_origin0,[1,num_of_points])
    
# tiling the origin1 to match the data    
    tiled_origin1= tf.tile(origin1, [tf.constant(1),num_of_points_in_each_batch])
    origin1_row = tf.reshape(tiled_origin1,[1,num_of_points])

# concatinating origin 0 and origin1 tiled 
    origin_2_rows = tf.concat([origin0_row,origin1_row],0)
    
# computing the translated and scaled xy
    xy_translated_scaled = tf.add(scaled_xy ,origin_2_rows) # 2 x N
    
         
    xy1 = tf.concat([xy_translated_scaled,row_of_ones],0)
    
    cen1 = tf.multiply(row_of_ones,cen[0])
    cen2 = tf.multiply(row_of_ones,cen[1])
    cen3 = tf.multiply(row_of_ones,cen[2])
    
    cen_mat = tf.concat([cen1,cen2],0)
    cen_mat = tf.concat([cen_mat,cen3],0)
    
    Rt_Ki = tf.matmul(Rt,Ki)
    Rt_Ki_xy1 = tf.matmul(Rt_Ki,xy1)
    
    point3D = tf.add(tf.multiply(Dlambda3,Rt_Ki_xy1),cen_mat)
    
    return point3D, batches

# *********************************************************************************************************

def dmap_to_nmap(Y, Rt, R, Ki, cen, Z, origin, scaling):
    BATCH_SIZE = tf.shape(Y)[0]
    IMAGE_HEIGHT = tf.shape(Y)[1]
    
    p3d, batches = depth2points3D_MB(Y, Rt, Ki, cen, Z, origin, scaling)
    
    p3d_map1 = tf.reshape(p3d[0,...],[BATCH_SIZE,IMAGE_HEIGHT,IMAGE_HEIGHT,1])
    p3d_map1 = tf.where(Z,p3d_map1,tf.zeros_like(p3d_map1))
    p3d_map2 = tf.reshape(p3d[1,...],[BATCH_SIZE,IMAGE_HEIGHT,IMAGE_HEIGHT,1])
    p3d_map2 = tf.where(Z,p3d_map2,tf.zeros_like(p3d_map2))
    p3d_map3 = tf.reshape(p3d[2,...],[BATCH_SIZE,IMAGE_HEIGHT,IMAGE_HEIGHT,1])
    p3d_map3 = tf.where(Z,p3d_map3,tf.zeros_like(p3d_map3))

    pcmap = tf.concat([p3d_map1,p3d_map2],3)
    pcmap = tf.concat([pcmap,p3d_map3],3)

    pcx_1 = tf.roll(pcmap, shift=-1, axis=2)
    pcy1 = tf.roll(pcmap, shift=1, axis=1)

    pcx_1_pc = pcx_1 - pcmap;
    pcy1_pc = pcy1 - pcmap;

    new_normal_map = cross(pcx_1_pc, pcy1_pc)
    n = new_normal_map
    output_mask = tf.abs(n) < 1e-5
    output_no0 = tf.where(output_mask, 1e-5*tf.ones_like(n), n)
    output_mag = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(output_no0),3)),-1)
    n = tf.truediv(output_no0,output_mag)

    n1 = n[...,0]
    n1 = tf.reshape(n1,[1,BATCH_SIZE*IMAGE_HEIGHT*IMAGE_HEIGHT])
    n2 = n[...,1]
    n2 = tf.reshape(n2,[1,BATCH_SIZE*IMAGE_HEIGHT*IMAGE_HEIGHT])
    n3 = n[...,2]
    n3 = tf.reshape(n3,[1,BATCH_SIZE*IMAGE_HEIGHT*IMAGE_HEIGHT])

    n_vec_all = tf.concat([n1,n2],0)
    n_vec_all = tf.concat([n_vec_all,n3],0)

    n_vec_all_rotated = tf.matmul(R,n_vec_all)

    n1v = n_vec_all_rotated[0,...]
    n1v = tf.reshape(n1v,[BATCH_SIZE,IMAGE_HEIGHT,IMAGE_HEIGHT,1])
    n2v = n_vec_all_rotated[1,...]
    n2v = tf.reshape(n2v,[BATCH_SIZE,IMAGE_HEIGHT,IMAGE_HEIGHT,1])
    n3v = n_vec_all_rotated[2,...]
    n3v = tf.reshape(n3v,[BATCH_SIZE,IMAGE_HEIGHT,IMAGE_HEIGHT,1])

    n_rotated = tf.concat([n1v,n2v],3)
    n_rotated = tf.concat([n_rotated,n3v],3)
    n = n_rotated
    output_mask = tf.abs(n) < 1e-5
    output_no0 = tf.where(output_mask, 1e-5*tf.ones_like(n), n)
    output_mag = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(output_no0),3)),-1)
    n = tf.truediv(output_no0,output_mag)
    
    return n
