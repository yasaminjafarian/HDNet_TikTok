import tensorflow as tf # tensorflow import
import numpy as np # python에서 벡터, 행렬 등 수치 연산을 수행하는 선형대수 라이브러리
import skimage.data # skimage는 이미지 처리하기 위한 파이썬 라이브러리
from PIL import Image, ImageDraw, ImageFont # PIL은 파이썬 인터프리터에 다양한 이미지 처리와 그래픽 기능을 제공하는 라이브러리
import math # 수학 관련 함수들이 들어있는 라이브러리
from tensorflow.python.platform import gfile # open()이랑 같고, tensorflow용 파일 입출력 함수
import scipy.misc # scipy에서 기타 함수 https://docs.scipy.org/doc/scipy/reference/misc.html
from utils.vector import cross #외적 함수

IMAGE_HEIGHT = 256 # 이미지 가로 크기
IMAGE_WIDTH = 256 # 이미지 세로 크기

# *****************************************************************************************************
#rigid_transform_3D(A,B): A와 B, 2개의 correspondence 좌표가 있을 때 그에 맞는 R과 T는 출력하는 함수
#get_pc_transformation2(p1,p2):rigid_transform_3D(p1, p2)사용 p1과 p2를 받으면 그것에 대한 R과 t, 그리고 받은 R과 t를 사용해서 예측한 p1_2를 output으로 내보낸다.
#Depth2Points3D_transformed_vector(Dlambda, indices , Rt, Ki, cen, origin, scaling): depth 정보(Dlambda), 미리 define한 점들의 index 정보인 indices에 x,y좌표 정보가 있고, camera intrinsic parameter Rt, Ki, cen, scaling factor 들을 받아서 3D reconstruction 점을 출력으로 내보낸다.
#part_transformation2(i_limit,PC1,PC2,p): get_pc_transformation2(p1, p2)을 사용해서 p1_2를 받고, 그 p2를 p2p로 저장해서 p2p와 p1_2를 보두 출력으로 뱉는다.
#transform_depth_PCs_dp_based2(C,R,Rt,cen,K,Ki,origin,scaling,d_i,d_j,i_r1_c1_r2_c2,i_limit): j번쨰 time instant의 실제 좌표와 warping function으로 예측한 좌표의 값을 출력하는 함수
#reproject(point3D, K,R,C): 3차원 좌표를 다시 2차원으로 reproject하는 것을 의미한다.
#compute_dp_tr_3d_2d_loss2(d_i,d_j,i_r1_c1_r2_c2,i_limit,C,R,Rt,cen,K,Ki,origin,scaling): 3차원 좌표끼리의 loss와 그 3차원 좌표를 reprojection한 좌표끼리의 loss를 모두 출력한다. 그리고 예측한 좌표와 실제 좌표도 출력한다.
# *****************************************************************************************************




def rigid_transform_3D(A,B):# B는 warping function에 의해 예측된 p값, A는 i번째 instance의 p값이다. 그리고 이 함수는 그에 맞는 R과 T를 내보낸다.
    A = tf.transpose(A) #3*N, tf.transpose는 matrix에 transpose를 시켜준다.
    B = tf.transpose(B) #3*N, B=R*A+T이다.
    num_rows = tf.shape(B)[0] #3, tf.shape는 input 텐서의 구조를 1-d 정수형 텐서로 반환한다. [0]은 행이다.
    num_cols = tf.shape(B)[1] #N, [1]은 열이다.
    centroid_A = tf.reshape(tf.reduce_mean(A,1),[3,1]) #3*1, 1*3을 3*1로 reshape했다.
    centroid_B = tf.reshape(tf.reduce_mean(B,1),[3,1]) #3*1
    one_row = tf.ones([1,num_cols], tf.float32) # 1*N, tf.ones는 모든 요소가 1로 설정된 텐서를 생성한다.
    Amean = tf.concat([one_row*centroid_A[0,0],one_row*centroid_A[1,0],one_row*centroid_A[2,0]],0) #3*N, 여기서 centoid_A[0,0]은 3*1 column의 첫번째 행의 값이다. 즉, 상수값이란 소리다. 그리고 tf.concat(,0)이기에 행을 위에서부터 이어붙이는 것이다.
    Bmean = tf.concat([one_row*centroid_B[0,0],one_row*centroid_B[1,0],one_row*centroid_B[2,0]],0) #3*N, 위의 코드 설명과 마찬가지
    
    Am = tf.subtract(A , Amean)#A의 각 행에 그 행의 평균을 모두 빼준다. 
    Bm = tf.subtract(B , Bmean)#B의 각 행에 그 행의 평균을 모두 빼준다.
    
    H = tf.matmul(Am , tf.transpose(Bm))#(3*N)*(N*3)=3*3
    
    S, U, V = tf.linalg.svd(H)#tf.linalg.svd(H)는 H를 SVD decomposition한다.
    R = tf.matmul(V,tf.transpose(U))#R을 구하는 식
    t = tf.matmul(R*(-1),centroid_A) + centroid_B#http://graphics.stanford.edu/~smr/ICP/comparison/eggert_comparison_mva97.pdf 에 자세히 나와있다. R과 T를 유도하는 식이다.
    return R,t

def get_pc_transformation2(p1,p2):#p1은 i번째 instance의 p값, p2는 warping function에 의해 예측된 j번째 instance의 p값
    R,t = rigid_transform_3D(p1, p2)#R과 t가 계산되어서 나온다.
    one_row = tf.ones([1,tf.shape(p1)[0]],tf.float32) # 1*N, p1이 N*3인데, tf.shape(p1)[0]은 row의 개수를 내보내기에 N을 내보내고, 1*N의 1로 구성된 matrix가 나온다. 
    tmat = tf.concat([one_row*t[0,0],one_row*t[1,0],one_row*t[2,0]],0) #3*N, t가 3*1인데, tmat의 첫 번째 row는 t의 첫번째 row값들이 N개 있고, 두번째 row는 t의 2번째 row값들이 N개 있고, 세번째 row는 t의 3번째 row값들이 N개 있다.
    p1_2 = tf.transpose(tf.matmul(R,tf.transpose(p1)) + tmat) #N*3, R*p1^T+tmat인데, (RA+T)^T이기에 N*3이 된다. 즉, 새로운 R과 T를 반영해서 계산한 warping function으로 예측된 j번째 instance의 p인 것 같다.
    return R,t, p1_2#새로운 R, t와 이것들을 반영하여 예측된 j번째 instance의 p값

# *****************************************************************************************************

def Depth2Points3D_transformed_vector(Dlambda, indices , Rt, Ki, cen, origin, scaling):#3D point를 reconstruction하는 함수
    
    num_of_points = tf.shape(Dlambda)[0] #N, Dlamda의 행의 shape을 정수형 tensor로 반환. Dlamda의 행이 N인가보다. 그리고 N은 point의 개수인 것 같다.
    num_of_batches = 1 #batch의 사이즈
    num_of_points_in_each_batch = tf.cast(tf.divide(num_of_points,num_of_batches),tf.int32)#tf.cast는 텐서를 새로운 형태로 캐스팅하는데 사용한다.tf.divide로 point의 개수를 batch의 사이즈로 나눠서 한 배치당 몇개의 point를 다루는지 출력한다.
    
    
    Dlambda_t = tf.reshape(Dlambda,[1,num_of_points]) # 1 x N, Dlamda가 N*1로 되어있는데, 이를 1*N으로 reshape하는 것 같다. 그냥 transpose와 같다고 볼 수 있을 것 같다.
    Dlambda3 = tf.concat([Dlambda_t,Dlambda_t],0)# 위 아래로 같은 행을 붙인 것이다. 2 x N
    Dlambda3 = tf.concat([Dlambda3,Dlambda_t],0) # 3 x N, 또 한 번 붙였다.
    
    
    
    idx = tf.cast(indices, tf.float32)# indices는 index들을 말하는 것 같다. tf.cat으로 이를 float32로 typecasting 한다.

    row_of_ones = tf.ones([1, num_of_points], tf.float32) # 1 x N의 요소가 모두 1로 이루어진 행렬 제작
    
# dividing xy and the batch number    
    bxy = idx # N x 3<--이라고 써있는데, N X 2인 것 같다.
    xy = tf.transpose(tf.reverse(bxy,[1])) # 2 x N, tf.reverse로 [1] 즉, 열을 반전시킨다. 그리고 나서 transpose를 한다.

# tiling the scaling to match the data(데이터를 일치시키기 위해 적도 조정)
    scaling2 = tf.reshape(scaling, [num_of_batches,1])# scaling값을 [1,1]로 reshape한다.
    tiled_scaling = tf.tile(scaling2, [tf.constant(1),num_of_points_in_each_batch])#tf.tile 함수는 주어진 텐서를 multiplies 만큼 이어붙이는 함수로 여기서는 1xN matrix가 된다.[scaling2,scaling2,....scaling2]
    scaling_row = tf.reshape(tiled_scaling,[1,num_of_points])#1XN, tiled scaling을 또 reshape한다.
    scaling_2_rows = tf.concat([scaling_row,scaling_row],0)#concat으로 이어붙여서 2xN이 된다.

# scaling the input 
    scaled_xy = tf.multiply(xy, scaling_2_rows)#2xN, tf.multifly는 각 요소별로 곱하는 것이다. scaling factor를 각 요소에 곱한다.

# dividing the origin 0 and origin 1 of the origin 
    origin0 = origin[...,0]#ixjxk에서 ixj는 모두 포함하고 k번째 index 기준으로 k번째 index가 0인 것을 가져오는 것
    origin0 = tf.reshape(origin0,[num_of_batches,1])#origin0을 [1,1]로 reshape한다.
    origin1 = origin[...,1]#ixjxk에서 ixj는 모두 포함하고 k번째 index 기준으로 k번째 index가 1인 것을 가져오는 것
    origin1 = tf.reshape(origin1,[num_of_batches,1])#origin1을 [1,1]로 reshape한다. 그냥 scalar인듯
    
# tiling the origin0 to match the data
    tiled_origin0= tf.tile(origin0, [tf.constant(1),num_of_points_in_each_batch])#1xN ,tf.tile 함수는 주어진 텐서를 multiplies 만큼 이어붙이는 함수로 여기서는 1xN matrix가 된다.[origin0,origin0,....origin0]
    origin0_row = tf.reshape(tiled_origin0,[1,num_of_points])#1xN인 것을 1xN으로 reshape한다. 결국 같다.
    
# tiling the origin1 to match the data    
    tiled_origin1= tf.tile(origin1, [tf.constant(1),num_of_points_in_each_batch])#1xN ,tf.tile 함수는 주어진 텐서를 multiplies 만큼 이어붙이는 함수로 여기서는 1xN matrix가 된다.[origin1,origin1,....origin1]
    origin1_row = tf.reshape(tiled_origin1,[1,num_of_points])#1xN인 것을 1xN으로 reshape한다. 결국 같다.

# concatinating origin 0 and origin1 tiled 
    origin_2_rows = tf.concat([origin0_row,origin1_row],0)#origin0_row와 origin1_row를 행으로 이어붙인다. 따라서 2xN이다.
    
# computing the translated and scaled xy
    xy_translated_scaled = tf.add(scaled_xy ,origin_2_rows) # 2 x N, scaled_xy와 origin을 요소별로 더한다. 이게 image 상의 point가 된다.
    
         
    xy1 = tf.concat([xy_translated_scaled,row_of_ones],0)#밑에가 모두 1인 3xN matrix인데 생각해보면 이게 homogeneus representation인 것 같다.
    
    cen1 = tf.multiply(row_of_ones,cen[0])#1xN인데 모든 요소가 cen[0]인 matrix
    cen2 = tf.multiply(row_of_ones,cen[1])#1xN인데 모든 요소가 cen[1]인 matrix
    cen3 = tf.multiply(row_of_ones,cen[2])#1xN인데 모든 요소가 cen[2]인 matrix
    
    cen_mat = tf.concat([cen1,cen2],0)
    cen_mat = tf.concat([cen_mat,cen3],0)#결국 1번째 행은 cen[0], 결국 2번째 행은 cen[1], 결국 3번째 행은 cen[2]인 3xN인 center matrix를 만든다.
    
    Rt_Ki = tf.matmul(Rt,Ki)#Rt는 그냥 identity matrix이고, Ki가 K 카메라 intrinsic camera parameter의 inverse matrix이다.
    Rt_Ki_xy1 = tf.matmul(Rt_Ki,xy1)#이건 그냥 그 image 좌표 매트릭스랑 카메라 인트린식 좌표 매트릭스랑 곱한거 
    
    point3D = tf.add(tf.multiply(Dlambda3,Rt_Ki_xy1),cen_mat)#3xN matrix이다. Dlamda3가 깊이인 것 같다. 그리거 모두 곱하고 cen_mat를 더해줘서 최종적으로 reconstruction한 3D point가 나온다.
    
     #DONE 
    
    return tf.transpose(point3D)# 3xN을 Nx3으로 transpose해서 출력한다.

# *****************************************************************************************************

def part_transformation2(i_limit,PC1,PC2,p):
    strp = i_limit[p,1]#part_transformation에서 미리 정의한 transformation을 part별로 대표하는 점들을 define한다고 했었는데, 해당 part p의 처음 point의 index인 것 같다.
    endp = i_limit[p,2]+1#part_transformation에서 미리 정의한 transformation을 part별로 대표하는 점들을 define한다고 했었는데, 해당 part p의 마지막 point의 index인 것 같다.
    p2p = tf.zeros([],dtype=tf.float32)#모든 요소가 0인 tensor를 정의하는 것이다.
    p1_2 = tf.zeros([],dtype=tf.float32)#모든 요소가 0인 tensor를 정의하는 것이다.
    p1 = PC1[strp:endp,:]#열은 모두 사용하고, 행은 strp번째에서 endp-1번째 행까지 사용한다.p1은 i번째 time instant의 3D 좌표값
    p2 = PC2[strp:endp,:]#열은 모두 사용하고, 행은 strp번째에서 endp-1번째 행까지 사용한다.p2은 warping function에 의해 예측된 j번째 time instant의 p값
    _,_,p1_2 = get_pc_transformation2(p1,p2)#새로운 R, t를 반영하여 예측된 j번째 instance의 p값을 p1_2라 하고, R과 t는 넘긴다.
    p2p = PC2[strp:endp,:]#열은 모두 사용하고, 행은 strp번째에서 endp-1번째 행까지 사용한다. 새로운 예측값이 아니라 과거의 예측값을 의미
    return p2p, p1_2#p2p는 과거의 예측값, p1_2는 새로운 R,t를 반영한 예측값

# *****************************************************************************************************

def transform_depth_PCs_dp_based2(C,R,Rt,cen,K,Ki,origin,scaling,d_i,d_j,i_r1_c1_r2_c2,i_limit):
    d1 = d_i[0,...,0]#i번째 time instant의 깊이 정보
    d2 = d_j[0,...,0]#j번째 time instant의 깊이 정보
    
    
    r1 = i_r1_c1_r2_c2[:,1]-1; c1 = i_r1_c1_r2_c2[:,2]-1;#r1은 i_r1_c1_r2_c2의 2번째 column, c1은 i_r1_c1_r2_c2의 3번째 column
    r2 = i_r1_c1_r2_c2[:,3]-1; c2 = i_r1_c1_r2_c2[:,4]-1;#r2은 i_r1_c1_r2_c2의 4번째 column, c2은 i_r1_c1_r2_c2의 5번째 column
    
    n = tf.shape(i_r1_c1_r2_c2)[0]#n은 i_r1_c1_r2_c2의 행의 개수
    r1 = tf.reshape(r1,[n,1]); c1 = tf.reshape(c1,[n,1]);#모두 nx1이었던 것을 다시 nx1로 reshape한다.
    r2 = tf.reshape(r2,[n,1]); c2 = tf.reshape(c2,[n,1]);#모두 nx1이었던 것을 다시 nx1로 reshape한다.
    

    indices1 = tf.concat([r1,c1],1)   #N*2 열로 붙인다.
    indices2 = tf.concat([r2,c2],1)   #N*2 열로 붙인다.

    lambda1 = tf.gather_nd(d1,indices1); #tf.gather_nd(params, indices, name=None),indices1에 따라 d1에서 값들을 모은다.
    lambda2 = tf.gather_nd(d2,indices2); #tf.gather_nd(params, indices, name=None),indices2에 따라 d2에서 값들을 모은다.
    
    PC1 = Depth2Points3D_transformed_vector(lambda1, indices1 , Rt, Ki, cen, origin, scaling)#i번째의 reconstruction한 3D coordinate parameter을 받는다. Nx3
    PC2 = Depth2Points3D_transformed_vector(lambda2, indices2 , Rt, Ki, cen, origin, scaling)#j번째의 reconstruction한 3D coordinate parameter을 받는다. Nx3
    
    PC2p, PC1_2 = part_transformation2(i_limit,PC1,PC2,0); #0번 part의 3D reconstruction 좌표들
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,1); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); #1번 part의 3D reconstruction 좌표들 그리고 tf.concat으로 행끼리 붙인다.
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,2); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); #2번 part의 3D reconstruction 좌표들 그리고 tf.concat으로 행끼리 붙인다.
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,3); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); #3번 part의 3D reconstruction 좌표들 그리고 tf.concat으로 행끼리 붙인다.
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,4); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); #4번 part의 3D reconstruction 좌표들 그리고 tf.concat으로 행끼리 붙인다.
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,5); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); #5번 part의 3D reconstruction 좌표들 그리고 tf.concat으로 행끼리 붙인다.
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,6); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); #6번 part의 3D reconstruction 좌표들 그리고 tf.concat으로 행끼리 붙인다.
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,7); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); #7번 part의 3D reconstruction 좌표들 그리고 tf.concat으로 행끼리 붙인다.
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,8); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); #8번 part의 3D reconstruction 좌표들 그리고 tf.concat으로 행끼리 붙인다.
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,9); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); #9번 part의 3D reconstruction 좌표들 그리고 tf.concat으로 행끼리 붙인다.
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,10); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); #10번 part의 3D reconstruction 좌표들 그리고 tf.concat으로 행끼리 붙인다.
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,11); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); #11번 part의 3D reconstruction 좌표들 그리고 tf.concat으로 행끼리 붙인다.
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,12); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); #12번 part의 3D reconstruction 좌표들 그리고 tf.concat으로 행끼리 붙인다.
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,13); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); #13번 part의 3D reconstruction 좌표들 그리고 tf.concat으로 행끼리 붙인다.
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,14); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); #14번 part의 3D reconstruction 좌표들 그리고 tf.concat으로 행끼리 붙인다.
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,15); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); #15번 part의 3D reconstruction 좌표들 그리고 tf.concat으로 행끼리 붙인다.
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,16); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); #16번 part의 3D reconstruction 좌표들 그리고 tf.concat으로 행끼리 붙인다.
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,17); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); #17번 part의 3D reconstruction 좌표들 그리고 tf.concat으로 행끼리 붙인다.
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,18); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); #18번 part의 3D reconstruction 좌표들 그리고 tf.concat으로 행끼리 붙인다.
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,19); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); #19번 part의 3D reconstruction 좌표들 그리고 tf.concat으로 행끼리 붙인다.
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,20); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); #20번 part의 3D reconstruction 좌표들 그리고 tf.concat으로 행끼리 붙인다.
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,21); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); #21번 part의 3D reconstruction 좌표들 그리고 tf.concat으로 행끼리 붙인다.
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,22); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); #22번 part의 3D reconstruction 좌표들 그리고 tf.concat으로 행끼리 붙인다.
    p2p, p1_2 = part_transformation2(i_limit,PC1,PC2,23); PC2p = tf.concat([PC2p,p2p],0); PC1_2 = tf.concat([PC1_2,p1_2],0); #23번 part의 3D reconstruction 좌표들 그리고 tf.concat으로 행끼리 붙인다.
    
    return PC2p, PC1_2 #최종적으로 모든 실제 j번째 instant의 3D 좌표, warping으로 예측한 3D 좌표가 모두 concat으로 저장되었다. 그걸 출력한다.

# *****************************************************************************************************

def reproject(point3D, K,R,C):
    # point3D is N*3 and M is 3*4
    # xy is N*2
    M = tf.matmul(K,R)
    M = tf.matmul(M,C)#결론적으로 M=KRC
    
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

