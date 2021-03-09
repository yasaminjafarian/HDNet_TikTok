import tensorflow as tf
import numpy as np
import skimage.data
from PIL import Image, ImageDraw, ImageFont
import math
from tensorflow.python.platform import gfile
import scipy.misc
import glob
import ntpath
from os import path


# **********************************************************************************************************
def write_matrix_txt(a,filename):
    mat = np.matrix(a)
    with open(filename,'wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.5f')
            
# **********************************************************************************************************
def get_origin_scaling(bbs, IMAGE_HEIGHT):
    Bsz = np.shape(bbs)[0]
    batch_origin = []
    batch_scaling = []
    
    for i in range(Bsz):
        bb1_t = bbs[i,...] - 1
        bbc1_t = bb1_t[2:4,0:3]
        
        origin = np.multiply([bb1_t[1,0]-bbc1_t[1,0],bb1_t[0,0]-bbc1_t[0,0]],2)

        squareSize = np.maximum(bb1_t[0,1]-bb1_t[0,0]+1,bb1_t[1,1]-bb1_t[1,0]+1);
        scaling = [np.multiply(np.true_divide(squareSize,IMAGE_HEIGHT),2)]
    
        batch_origin.append(origin)
        batch_scaling.append(scaling)
    
    batch_origin = np.array(batch_origin,dtype='f')
    batch_scaling = np.array(batch_scaling,dtype='f')
    
    O = np.zeros((Bsz,1,2),dtype='f')
    O = batch_origin
    
    S = np.zeros((Bsz,1),dtype='f')
    S = batch_scaling
    
    return O, S

# **********************************************************************************************************
def read_test_data(data_main_path,data_name,IMAGE_HEIGHT,IMAGE_WIDTH):
    image_path = data_main_path +"/" + data_name + "_img.png"
    mask_path = data_main_path +"/" + data_name + "_mask.png"
    dp_path = data_main_path +"/" + data_name + "_dp.png"
    
    color = np.array(scipy.misc.imread(image_path),dtype='f')
    mask = np.array(scipy.misc.imread(mask_path),dtype='f')
    dp = np.array(scipy.misc.imread(dp_path),dtype='f')
    
    X = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')
    X[0,...] = color
    
    Z = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,1),dtype='f')
    Z[0,...,0] = mask>100
    
    DP = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')
    DP[0,...] = dp
    
    Z2C3 = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='b')
    Z2C3[...,0]=Z[...,0]
    Z2C3[...,1]=Z[...,0]
    Z2C3[...,2]=Z[...,0]
    
    X = np.where(Z2C3,X,np.ones_like(X)*255.0)
    
    Z3 = Z2C3
    
    # camera
    C = np.zeros((3,4),dtype='f')
    C[0,0]=1
    C[1,1]=1
    C[2,2]=1
    
    R = np.zeros((3,3),dtype='f')
    R[0,0]=1
    R[1,1]=1
    R[2,2]=1
    
    Rt = R
    
    K = np.zeros((3,3),dtype='f')
    K[0,0]=1111.6
    K[1,1]=1111.6

    K[0,2]=960
    K[1,2]=540
    K[2,2]=1
    
    Ki = np.linalg.inv(K)
    cen = np.zeros((3),dtype='f')
    bbs = np.array([[25,477],[420,872],[1,453],[1,453]],dtype='f')
    bbs = np.reshape(bbs,[1,4,2])
    (origin, scaling) = get_origin_scaling(bbs, IMAGE_HEIGHT)
    
    return X,Z, Z3, C, cen,K,Ki, R,Rt,  scaling, origin, DP

# **********************************************************************************************************
def nmap_normalization(nmap_batch):
    image_mag = np.expand_dims(np.sqrt(np.square(nmap_batch).sum(axis=3)),-1)
    image_unit = np.divide(nmap_batch,image_mag)
    return image_unit
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst    
# **********************************************************************************************************   
def path_leaf(inpath):
    head, tail = ntpath.split(inpath)
    return tail or ntpath.basename(head)
def get_test_data(inpath):
    pngpath = inpath+'/*_img.png'
    all_img  = glob.glob(pngpath)
    filename_list = []
    for i in range(len(all_img)):
        img_name = path_leaf(all_img[i])
        name = img_name[0:-8]
        dpname = name+"_dp.png"
        mname = name+"_mask.png"
        if path.exists(inpath+'/'+dpname) and path.exists(inpath+'/'+mname):
            filename_list.append(name)          
    return filename_list
# **********************************************************************************************************   
def depth2mesh(depth, mask, filename):
    h = depth.shape[0]
    w = depth.shape[1]
    depth = depth.reshape(h,w,1)
    f = open(filename + ".obj", "w")
    for i in range(h):
        for j in range(w):
            f.write('v '+str(float(2.0*i/h))+' '+str(float(2.0*j/w))+' '+str(float(depth[i,j,0]))+'\n')

    threshold = 0.07

    for i in range(h-1):
        for j in range(w-1):
            if i < 2 or j < 2:
                continue
            localpatch= np.copy(depth[i-1:i+2,j-1:j+2])
            dy_u = localpatch[0,:] - localpatch[1,:]
            dx_l = localpatch[:,0] - localpatch[:,1]
            dy_d = localpatch[0,:] - localpatch[-1,:]
            dx_r = localpatch[:,0] - localpatch[:,-1]
            dy_u = np.abs(dy_u)
            dx_l = np.abs(dx_l)
            dy_d = np.abs(dy_d)
            dx_r = np.abs(dx_r)
            if np.max(dy_u)<threshold and np.max(dx_l) < threshold and np.max(dy_d) < threshold and np.max(dx_r) < threshold and mask[i,j]:
                f.write('f '+str(int(j+i*w+1))+' '+str(int(j+i*w+1+1))+' '+str(int((i + 1)*w+j+1))+'\n')
                f.write('f '+str(int((i+1)*w+j+1+1))+' '+str(int((i+1)*w+j+1))+' '+str(int(i * w + j + 1 + 1)) + '\n')
    f.close()
    return