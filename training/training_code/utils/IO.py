import tensorflow as tf #텐서플로우 라이브러리
import numpy as np      #numpy 라이브러리
import skimage.data     #이미지처리에 특화된 Python 이미지 라이브러리, Numpy배열로 동작해서 이미지 객체를 처리함/
from PIL import Image, ImageDraw, ImageFont
import math
from tensorflow.python.platform import gfile
import scipy.misc
import matplotlib.pyplot as plt
import os.path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

## ******************************Write prediction************************************

def write_matrix_txt(a,filename):   #입력된 배열과 파일 이름을 가지고 행렬로 바꾼 뒤 txt파일로 저장해줌. test_data -> infer.out의 txt파일들이 이를 이용해 만들어진다.
    mat = np.matrix(a)  #a 입력으로 list입력들어오면 배열로 바꿔줌. a=[[0, 1, 2], [3, 4, 5]] 꼴 가능. 2x3 array
    with open(filename,'wb') as f:  #b는 파일을 binary mode로 연다. w는 쓰기모드 //with open(파일 경로, 모드) as 파일 객체: with as구문을 빠져나가게 되면 자동으로 close()함수 호출해 파일 닫음.
        for line in mat:            #mat의 element개수만큼 반복, line이 변수. 확실하지 않음.
            np.savetxt(f, line, fmt='%.5f')     #텍스트파일로 저장, f: 파일이름이나 파일handle, line: txt로 저장될 데이터. fmt:포맷.소수점 아래 5자리까지 표시.
            
def write_prediction(Vis_dir,prediction,i,idx,Z_r1):    #예측을 작성?
    image_tensor  = np.asarray(prediction)      #이미지 텐서. prediction으로 뭐가 들어오는지 파악 못함. asarray는 입력을 array로 바꿔주는 array와 동일 기능. but 데이터타입 옵션을 주면 동일한 데이터타입이어야 카피가 됨.
    num = 0  
    image = image_tensor[num,...,0] #이미지 = (0...0) 배열
    mask = Z_r1[num,...,0]>0        #
    write_matrix_txt(image,Vis_dir+"STEP%07d_frame%07d_DEPTH.txt" % (i,idx[0]))     #Vis_dir = test_data->infer_out폴더 (?), 이미지 배열을 
    
def write_prediction_normal(Vis_dir,prediction,i,idx,Z_r1):
    image_tensor  = np.asarray(prediction)  #prediction 이라는 내용의 배열 생성
    image = image_tensor[0,...] #이미지 = 이미지 텐서(배열)
    image_mag = np.expand_dims(np.sqrt(np.square(image).sum(axis=2)),-1)    #이미지_mag = root(이미지 각 배열요소 제곱하고 z축에 대해 합함.) y축에 대해 dimension 추가 -1, 1 동일 의미
    image_unit = np.divide(image,image_mag) #divide는 요소별 나눗셈 실행
    write_matrix_txt(image_unit[...,0],Vis_dir+"STEP%07d_frame%07d_NORMAL_1.txt" % (i,idx[0]))  #infer_out 폴더 안에 STEP 폴더? 없지 않나 %07d = 0을 7개 채우고 i, idx[0]을 오른쪽부터 채워넣음
    write_matrix_txt(image_unit[...,1],Vis_dir+"STEP%07d_frame%07d_NORMAL_2.txt" % (i,idx[0]))  #
    write_matrix_txt(image_unit[...,2],Vis_dir+"STEP%07d_frame%07d_NORMAL_3.txt" % (i,idx[0]))

# **********************************************************************************************************
def nmap_normalization(nmap_batch): #nmap normalization
    image_mag = np.expand_dims(np.sqrt(np.square(nmap_batch).sum(axis=2)),-1)   #
    image_unit = np.divide(nmap_batch,image_mag)
    return image_unit  #이게 되나?

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst  

def save_prediction_png (image,imagen,X,Z,Z3,Vis_dir,i,idx,perc):
    imagen=nmap_normalization(imagen)
    data_name = "salam"
    depth_map = image*Z[0,...,0]
    normal_map = imagen*Z3[0,...]
    min_depth = np.amin(depth_map[depth_map>0])
    max_depth = np.amax(depth_map[depth_map>0])*perc
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
    final_im.save(Vis_dir+"STEP%07d_frame%07d_results.png" % (i,idx[0]))
    os.remove(Vis_dir+data_name+"_depth.png")
    os.remove(Vis_dir+data_name+"_normal.png")
    
def save_prediction_png_normal (imagen,X,Z,Z3,Vis_dir,i,idx):
    imagen = nmap_normalization(imagen)
    data_name = "salam"
    normal_map = imagen*Z3[0,...]
    normal_map_rgb = -1*normal_map
    normal_map_rgb[...,2] = -1*((normal_map[...,2]*2)+1)
    normal_map_rgb = np.reshape(normal_map_rgb, [256,256,3]);
    normal_map_rgb = (((normal_map_rgb + 1) / 2) * 255).astype(np.uint8);
    plt.imsave(Vis_dir+data_name+"_normal.png", normal_map_rgb)   
    n = np.array(scipy.misc.imread(Vis_dir+data_name+"_normal.png"),dtype='f')
    n = np.where(Z3[0,...]>0,n[...,0:3],255.0)
    final_im = get_concat_h(Image.fromarray(np.uint8(X[0,...])),Image.fromarray(np.uint8(n)))
    final_im.save(Vis_dir+"STEP%07d_frame%07d_results.png" % (i,idx[0]))
    os.remove(Vis_dir+data_name+"_normal.png")

## *********************************Read RP trainign data**********************************************
    
max_corr_points = 25000
def read_renderpeople(rp_path, frms, IMAGE_HEIGHT,IMAGE_WIDTH):
    Bsize = len(frms)
    batch_densepose = []
    batch_color = []    
    batch_mask  = []
    batch_depth = []
    batch_normal = []
    for b in range(Bsize):
        cur_f = int(frms[b])
        name = "%07d" %(cur_f)
        batch_densepose.append(scipy.misc.imread(rp_path +'/densepose/'+name+'.png'))
        batch_color.append(scipy.misc.imread(rp_path +'/color_WO_bg/'+name+'.png'))
        batch_mask.append(scipy.misc.imread(rp_path +'/mask/'+name+'.png'))
        batch_depth.append(np.genfromtxt(rp_path +'/depth/'+name+'.txt',delimiter=","))
        cur_normal = np.array(scipy.misc.imread(rp_path +'/normal_png/'+name+'.png'),dtype='f');
        n1 = np.genfromtxt(rp_path +'/normal/'+name+'_1.txt',delimiter=",")
        n2 = np.genfromtxt(rp_path +'/normal/'+name+'_2.txt',delimiter=",")
        n3 = np.genfromtxt(rp_path +'/normal/'+name+'_3.txt',delimiter=",")
        cur_normal[...,0] = n1;
        cur_normal[...,1] = n2;
        cur_normal[...,2] = n3;
        batch_normal.append(cur_normal)
    
    batch_color = np.array(batch_color,dtype='f')
    batch_mask = np.array(batch_mask,dtype='f')
    batch_depth = np.array(batch_depth,dtype='f')
    batch_normal = np.array(batch_normal,dtype='f')
    batch_densepose = np.array(batch_densepose,dtype='f')
    
    X1 = np.zeros((Bsize,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')
    X1 = batch_color
    Y1 = np.zeros((Bsize,IMAGE_HEIGHT,IMAGE_WIDTH,1),dtype='f')
    Y1[...,0] = batch_depth
    Z1 = np.zeros((Bsize,IMAGE_HEIGHT,IMAGE_WIDTH,1),dtype='b')
    Z1[...,0] = batch_mask > 100
    Z1[Y1<1.0] = False
    Z1[Y1>8.0] = False
    N1 = np.zeros((Bsize,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')
    N1 = batch_normal
    DP1 = np.zeros((Bsize,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')
    DP1 = batch_densepose
    
    Z1_3 = np.zeros((Bsize,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='b')
    Z1_3[...,0]=Z1[...,0]
    Z1_3[...,1]=Z1[...,0]
    Z1_3[...,2]=Z1[...,0]
    
    # make the image with white background.
    X1 = np.where(Z1_3,X1,np.ones_like(X1)*255.0)
    N1 = np.where(Z1_3,N1,np.zeros_like(N1))
    Y1 = np.where(Z1,Y1,np.zeros_like(Y1))
    
    # shift the depthmap to median 4.
    Y2 = Y1
    for b in range(Bsize):
        yt = Y1[b,...]
        yt_n0 = yt[yt>0]
        med_yt = np.median(yt_n0)
        yt = yt + 4 - med_yt
        Y2[b,...] = yt
    Y1 = Y2
    
    Y1 = np.where(Z1,Y1,np.zeros_like(Y1))
    
    X_1 = np.zeros((Bsize,IMAGE_HEIGHT,IMAGE_WIDTH,9),dtype='f')

    X_1[...,0]=X1[...,0]
    X_1[...,1]=X1[...,1]
    X_1[...,2]=X1[...,2]
    X_1[...,3]=N1[...,0]
    X_1[...,4]=N1[...,1]
    X_1[...,5]=N1[...,2]
    X_1[...,6]=DP1[...,0]
    X_1[...,7]=DP1[...,1]
    X_1[...,8]=DP1[...,2]
    
    return X_1, X1, Y1, N1, Z1, DP1, Z1_3

def get_renderpeople_patch(rp_path,Bsize,image_nums,IMAGE_HEIGHT,IMAGE_WIDTH):
    num_of_ims = len(image_nums)
    frms_nums = np.random.choice(num_of_ims, Bsize).tolist()
    frms= []
    for f in range(len(frms_nums)):
        frms = frms + [image_nums[frms_nums[f]]]
    X_1, X1, Y1, N1, Z1, DP1, Z1_3 = read_renderpeople(rp_path, frms, IMAGE_HEIGHT,IMAGE_WIDTH)
    return X_1, X1, Y1, N1, Z1, DP1, Z1_3, frms

## *********************************Read Tk trainign data**********************************************
def read_tiktok(tk_path, frms, IMAGE_HEIGHT,IMAGE_WIDTH):
    Bsize = len(frms)
    batch_densepose = []
    batch_color = []    
    batch_mask  = []
    batch_depth = []
    batch_normal = []
    for b in range(Bsize):
        cur_f = int(frms[b])
        name = "%07d" %(cur_f)
        batch_densepose.append(scipy.misc.imread(tk_path +'/densepose/'+name+'.png'))
        batch_color.append(scipy.misc.imread(tk_path +'/color_WO_bg/'+name+'.png'))
        batch_mask.append(scipy.misc.imread(tk_path +'/mask/'+name+'.png'))
        cur_normal = np.array(scipy.misc.imread(tk_path +'/pred_normals_png/'+name+'.png'),dtype='f');
        n1 = np.genfromtxt(tk_path +'/pred_normals/'+name+'_1.txt',delimiter=" ")
        n2 = np.genfromtxt(tk_path +'/pred_normals/'+name+'_2.txt',delimiter=" ")
        n3 = np.genfromtxt(tk_path +'/pred_normals/'+name+'_3.txt',delimiter=" ")
        cur_normal[...,0] = n1;
        cur_normal[...,1] = n2;
        cur_normal[...,2] = n3;
        batch_normal.append(cur_normal)  
    batch_color = np.array(batch_color,dtype='f')
    batch_mask = np.array(batch_mask,dtype='f')
    batch_normal = np.array(batch_normal,dtype='f')
    batch_densepose = np.array(batch_densepose,dtype='f')
    
    X1 = np.zeros((Bsize,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')
    X1 = batch_color
    Z1 = np.zeros((Bsize,IMAGE_HEIGHT,IMAGE_WIDTH,1),dtype='b')
    Z1[...,0] = batch_mask > 100
    N1 = np.zeros((Bsize,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')
    N1 = batch_normal
    DP1 = np.zeros((Bsize,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')
    DP1 = batch_densepose
    
    Z1_3 = np.zeros((Bsize,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='b')
    Z1_3[...,0]=Z1[...,0]
    Z1_3[...,1]=Z1[...,0]
    Z1_3[...,2]=Z1[...,0]
    
    # make the image with white background.
    X1 = np.where(Z1_3,X1,np.ones_like(X1)*255.0)
    N1 = np.where(Z1_3,N1,np.zeros_like(N1))
    
    X_1 = np.zeros((Bsize,IMAGE_HEIGHT,IMAGE_WIDTH,9),dtype='f')

    X_1[...,0]=X1[...,0]
    X_1[...,1]=X1[...,1]
    X_1[...,2]=X1[...,2]
    X_1[...,3]=N1[...,0]
    X_1[...,4]=N1[...,1]
    X_1[...,5]=N1[...,2]
    X_1[...,6]=DP1[...,0]
    X_1[...,7]=DP1[...,1]
    X_1[...,8]=DP1[...,2]
    
    return X_1, X1, N1, Z1, DP1, Z1_3

def read_correspondences_dp(f,fi,corr_path):
    
    name_i = '%07d'%(f)
    name_j = '%07d'%(fi)
    
    i_r1_c1_r2_c2n = np.array(np.genfromtxt(corr_path+'corrs/'+name_i+'_'+name_j+'_i_r1_c1_r2_c2.txt',delimiter=","))
    i_r1_c1_r2_c2n = i_r1_c1_r2_c2n.astype('int')
    
    i_r1_c1_r2_c2n_f = np.zeros((max_corr_points,5),dtype='int')
    i_r1_c1_r2_c2n_f[0:i_r1_c1_r2_c2n.shape[0],:]=i_r1_c1_r2_c2n

    i_limitn = np.array(np.genfromtxt(corr_path+'corrs/'+name_i+'_'+name_j+'_i_limit.txt',delimiter=","))
    i_limitn = i_limitn.astype('int')
    return i_r1_c1_r2_c2n_f,i_limitn

def get_tiktok_patch(tk_path, Bsize, IMAGE_HEIGHT,IMAGE_WIDTH):#titok data를 사용하려고 한다.
    corr_mat = np.genfromtxt(tk_path +'/correspondences/corr_mat.txt',delimiter=",")#79x5 matrix, np.genfromtxt(,delimiter=",")을 사용하여 "," 구분이 열이고, enter가 행으로 matrix를 받아온다.
    corr_path = tk_path +'/correspondences/'#
    num_of_neighbors = np.shape(corr_mat)[1]-1#일단 열이 5이기에 5-1=4
    image_nums = corr_mat[:,0].tolist()#list로 바꿔주는 함수인 tolist() 덕분에 일단 첫번째 
    num_of_ims = len(image_nums)
    frms_nums = np.random.choice(num_of_ims, Bsize).tolist()
    frms= []
    frms_neighbor = []
    bi_r1_c1_r2_c2 = []
    bi_limit = []
    for f in range(len(frms_nums)):
        row = frms_nums[f]
        frm = image_nums[row]
        frms = frms + [frm]
        
        neighbor_choice = np.random.choice(num_of_neighbors, 1)[0] + 1
        nfrm = corr_mat[row,neighbor_choice]
        frms_neighbor = frms_neighbor + [nfrm]
        
        i_r1_c1_r2_c2n_f,i_limitn = read_correspondences_dp(frm,nfrm,corr_path)
        bi_r1_c1_r2_c2.append(i_r1_c1_r2_c2n_f)
        bi_limit.append(i_limitn)
        
    i_r1_c1_r2_c2 = np.zeros((Bsize,max_corr_points,5),dtype='i')
    i_r1_c1_r2_c2 = bi_r1_c1_r2_c2
    i_limit = np.zeros((Bsize,24,3),dtype='i')
    i_limit = bi_limit
    
    X_1, X1, N1, Z1, DP1, Z1_3 = read_tiktok(tk_path, frms, IMAGE_HEIGHT,IMAGE_WIDTH)
    X_2, X2, N2, Z2, DP2, Z2_3 = read_tiktok(tk_path, frms_neighbor, IMAGE_HEIGHT,IMAGE_WIDTH)
    
    return X_1, X1, N1, Z1, DP1, Z1_3, X_2, X2, N2, Z2, DP2, Z2_3, i_r1_c1_r2_c2, i_limit, frms, frms_neighbor
    
## **************************************Get Camera**********************************************
def get_origin_scaling(bbs, IMAGE_HEIGHT):#bbs는 Bsizex4x2인 float형 0으로 이루어진 matrix이다. 그리고 IMAGE_HEIGHT를 입력으로 받는다.
    Bsz = np.shape(bbs)[0]#일단 첫번째 축의 크기를 Bsz에 저장하는데 그냥 batchsize다.
    batch_origin = []#빈 array를 선언한다.
    batch_scaling = []#빈 array를 선언한다.
    
    for i in range(Bsz):
        bb1_t = bbs[i,...] - 1#i번째 batch의 값을 저장한다. 4x2이다. 그리고 모든 요소에서 1을 뺀다.
        bbc1_t = bb1_t[2:4,0:3]#2x2이다. 4개의 행 중에 밑에 행 2개를 bbc1_t에 따로 저장한다.
        
        origin = np.multiply([bb1_t[1,0]-bbc1_t[1,0],bb1_t[0,0]-bbc1_t[0,0]],2)#[2*(bb1_t[1,0]-bbc1_t[1,0]),2*(bb1_t[0,0]-bbc1_t[0,0])]

        squareSize = np.maximum(bb1_t[0,1]-bb1_t[0,0]+1,bb1_t[1,1]-bb1_t[1,0]+1);#가장 큰 값을 정사각형의 한 변으로 한다.
        scaling = [np.multiply(np.true_divide(squareSize,IMAGE_HEIGHT),2)]#(squareSize/IMAGE_HEIGHT)*2
    
        batch_origin.append(origin)#append() 때문에 계속 [x, y]를 batch_origin에 넣어서 [[x,y],[],[],[]..]가 된다.
        batch_scaling.append(scaling)#[[],[],[],[],[]...]이것도 append 때문에 계속 list에 [scaling]을 넣는다.
    
    batch_origin = np.array(batch_origin,dtype='f')#각각을 다시 float형 array로 만들어서 저장한다.
    batch_scaling = np.array(batch_scaling,dtype='f')#
    
    O = np.zeros((Bsz,1,2),dtype='f')
    O = batch_origin
    
    S = np.zeros((Bsz,1),dtype='f')
    S = batch_scaling
    
    return O, S

def get_camera(Bsize,IMAGE_HEIGHT):#일단 IMAGE_HEIGHT와 batch size를 input으로 한다. batch size는 주로 1 또는 8이다.
    C1n = np.zeros((3,4),dtype='f')#data type float로 3x4인 0으로 이루어진 행렬을 만든다.
    C1n[0,0]=1#그리고 각 행렬의 (0,0),(1,1),(2,2)에 1을 넣는다.
    C1n[1,1]=1
    C1n[2,2]=1
    #       1 0 0 0
    # C1n = 0 1 0 0
    #       0 0 1 0
    R1n = np.zeros((3,3),dtype='f')#data type float로 3x3인 0으로 이루어진 행렬을 만든다.
    R1n[0,0]=1#그리고 각 행렬의 (0,0),(1,1),(2,2)에 1을 넣는다.
    R1n[1,1]=1
    R1n[2,2]=1
    
    Rt1n = R1n
    #       1 0 0
    # R1n = 0 1 0
    #       0 0 1
    K1n = np.zeros((3,3),dtype='f')#data type float로 3x3인 0으로 이루어진 행렬을 만든다.K1n은 camera intrinsic parameter이다.
    
    K1n[0,0]=1111.6
    K1n[1,1]=1111.6

    K1n[0,2]=960
    K1n[1,2]=540
    K1n[2,2]=1
    #       1111.6      0 960
    # R1n =      0 1111.6 540
    #            0      0   1
    M1n = np.matmul(np.matmul(K1n,R1n),C1n)#3x4 행렬인데, 마지막 column이 0이고, 앞의 3 column은 K1n과 같다.

    Ki1n = np.linalg.inv(K1n)#K1n matrix를 inverse 시킨 matrix인데, 우리가 3D point reconstruction할 때는 inverse가 필요하기에 inverse한것을 Ki1n에 저장하는 것 같다.
    
    cen1n = np.zeros((3),dtype='f')#1x3의 모든 요소가 float형 0으로 이루어진 matrix
    
    bbs1n_tmp = np.array([[25,477],[420,872],[1,453],[1,453]],dtype='f')#float형이고, 4x2인 matrix이다.흠..이건 bounding box인가?
    bbs1n_tmp = np.reshape(bbs1n_tmp,[1,4,2])#reshape으로 앞에 한 axis를 추가하여 1x4x2이다.
    
    bbs1n = np.zeros((Bsize,4,2),dtype='f')# Bsizex4x2인 float형 0으로 이루어진 matrix이다.
    for b in range(Bsize):#Bsizex4x2인데 각 batch마다 bbs1n_tmp를 모두 저장한다.
        bbs1n[b,...]=bbs1n_tmp[0,...]
           
    origin1n, scaling1n = get_origin_scaling(bbs1n, IMAGE_HEIGHT)
    
    return origin1n, scaling1n, C1n, cen1n, K1n, Ki1n, M1n, R1n, Rt1n