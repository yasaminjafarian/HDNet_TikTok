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


def rigid_transform_3D(A,B):
    A = tf.transpose(A) #3*N
    B = tf.transpose(B) #3*N
    num_rows = tf.shape(B)[0] #3
    num_cols = tf.shape(B)[1] #N
    centroid_A = tf.reshape(tf.reduce_mean(A,1),[3,1]) #3*1
    centroid_B = tf.reshape(tf.reduce_mean(B,1),[3,1]) #3*1
    one_row = tf.ones([1,num_cols], tf.float32) # 1*N
    Amean = tf.concat([one_row*centroid_A[0,0],one_row*centroid_A[1,0],one_row*centroid_A[2,0]],0) #3*N
    Bmean = tf.concat([one_row*centroid_B[0,0],one_row*centroid_B[1,0],one_row*centroid_B[2,0]],0) #3*N
    
    Am = tf.subtract(A , Amean)
    Bm = tf.subtract(B , Bmean)
    
    H = tf.matmul(Am , tf.transpose(Bm))
    
    S, U, V = tf.linalg.svd(H)
    R = tf.matmul(V,tf.transpose(U))
    t = tf.matmul(R*(-1),centroid_A) + centroid_B
    return R,t

def get_pc_transformation2(p1,p2):
    R,t = rigid_transform_3D(p1, p2)
    one_row = tf.ones([1,tf.shape(p1)[0]],tf.float32) # 1*N
    tmat = tf.concat([one_row*t[0,0],one_row*t[1,0],one_row*t[2,0]],0) #3*N
    p1_2 = tf.transpose(tf.matmul(R,tf.transpose(p1)) + tmat) #N*3
    return R,t, p1_2

# *****************************************************************************************************

def Depth2Points3D_transformed_vector(Dlambda, indices , Rt, Ki, cen, origin, scaling):
    
    num_of_points = tf.shape(Dlambda)[0] #N
    num_of_batches = 1
    num_of_points_in_each_batch = tf.cast(tf.divide(num_of_points,num_of_batches),tf.int32)
    
    
    Dlambda_t = tf.reshape(Dlambda,[1,num_of_points]) # 1 x N
    Dlambda3 = tf.concat([Dlambda_t,Dlambda_t],0)
    Dlambda3 = tf.concat([Dlambda3,Dlambda_t],0) # 3 x N
    
    
    
    idx = tf.cast(indices, tf.float32)

    row_of_ones = tf.ones([1, num_of_points], tf.float32) # 1 x N
    
# dividing xy and the batch number    
    bxy = idx # N x 3
    xy = tf.transpose(tf.reverse(bxy,[1])) # 2 x N

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
    
     #DONE 
    
    return tf.transpose(point3D)

# *****************************************************************************************************

def part_transformation2(i_limit,PC1,PC2,p):
    strp = i_limit[p,1]
    endp = i_limit[p,2]+1
    p2p = tf.zeros([],dtype=tf.float32)
    p1_2 = tf.zeros([],dtype=tf.float32)
    p1 = PC1[strp:endp,:]
    p2 = PC2[strp:endp,:]
    _,_,p1_2 = get_pc_transformation2(p1,p2)
    p2p = PC2[strp:endp,:]
    return p2p, p1_2

# *****************************************************************************************************

def transform_depth_PCs_dp_based2(C,R,Rt,cen,K,Ki,origin,scaling,d_i,d_j,i_r1_c1_r2_c2,i_limit):
    d1 = d_i[0,...,0]
    d2 = d_j[0,...,0]
    
    
    r1 = i_r1_c1_r2_c2[:,1]-1; c1 = i_r1_c1_r2_c2[:,2]-1;
    r2 = i_r1_c1_r2_c2[:,3]-1; c2 = i_r1_c1_r2_c2[:,4]-1;
    
    n = tf.shape(i_r1_c1_r2_c2)[0]
    r1 = tf.reshape(r1,[n,1]); c1 = tf.reshape(c1,[n,1]);
    r2 = tf.reshape(r2,[n,1]); c2 = tf.reshape(c2,[n,1]);
    

    indices1 = tf.concat([r1,c1],1)   #N*2
    indices2 = tf.concat([r2,c2],1) 

    lambda1 = tf.gather_nd(d1,indices1); 
    lambda2 = tf.gather_nd(d2,indices2); 
    
    PC1 = Depth2Points3D_transformed_vector(lambda1, indices1 , Rt, Ki, cen, origin, scaling)
    PC2 = Depth2Points3D_transformed_vector(lambda2, indices2 , Rt, Ki, cen, origin, scaling)
    
    PC2p, PC1_2 = part_transformation2(i_limit,PC1,PC2,0);
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,1); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); 
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,2); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); 
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,3); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); 
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,4); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); 
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,5); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); 
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,6); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); 
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,7); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); 
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,8); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); 
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,9); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); 
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,10); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); 
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,11); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); 
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,12); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); 
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,13); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); 
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,14); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); 
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,15); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); 
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,16); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); 
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,17); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); 
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,18); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); 
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,19); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); 
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,20); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); 
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,21); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); 
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,22); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); 
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,23); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); 
    
    return PC2p, PC1_2

# *****************************************************************************************************

def reproject(point3D, K,R,C):
    # point3D is N*3 and M is 3*4
    # xy is N*2
    M = tf.matmul(K,R)
    M = tf.matmul(M,C)
    
    point3D = tf.transpose(point3D)
    
    num_of_points = tf.shape(point3D)[1]
    
    row_of_ones = tf.ones([1, num_of_points], tf.float32)
    
    xyz1 = tf.concat([point3D,row_of_ones],0)
    
    xyS = tf.matmul(M, xyz1)
    
    S = xyS[2,...]
    S = tf.reshape(S,[1,num_of_points])
    S2 = tf.concat([S,S],0)
    S3 = tf.concat([S2,S],0)
    
    xy1 = tf.truediv(xyS, S3)
    
    xy = xy1[0:2,...]
    
    xy = tf.transpose(xy)
    
    x = xy[...,0]; x=tf.reshape(x,[num_of_points,1])
    y = xy[...,1]; y=tf.reshape(y,[num_of_points,1])
    
    rc = tf.concat([y,x],1)
    
    return xy,rc

# *****************************************************************************************************

def compute_dp_tr_3d_2d_loss2(d_i,d_j,i_r1_c1_r2_c2,i_limit,C,R,Rt,cen,K,Ki,origin,scaling):
    PC2p, PC1_2 = transform_depth_PCs_dp_based2(C,R,Rt,cen,K,Ki,origin,scaling,d_i,d_j,i_r1_c1_r2_c2,i_limit)
    
    d = tf.subtract(PC2p, PC1_2)
    err_vec = tf.sqrt(tf.reduce_sum(tf.square(d),1));
    loss3d = tf.reduce_mean(err_vec)
    
    x2,_ = reproject(PC2p, K,R,C)
    x1_2,_ = reproject(PC1_2, K,R,C)
    
    d = tf.subtract(x2, x1_2)
    err_vec = tf.sqrt(tf.reduce_sum(tf.square(d),1));
    loss2d = tf.reduce_mean(err_vec)
    
    return loss3d, loss2d,PC2p, PC1_2

# *****************************************************************************************************

