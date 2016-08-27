import numpy as np
import sys
import cv2
import math
import copy
import pdb
import time
import numpy.matlib as npmat
import copy
from get_data import horizon
import matplotlib.pyplot as plt
import skcuda.misc as misc

from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
import pycuda.driver as drv
#from show_result_v3 import visualize

start= time.time()

print '%40s' % '---------------------begining--------------------------'
ID= sys.argv[1]
print 'ID=',ID,'is been proccessing'
k=float(sys.argv[2])
print('k',k)
alpha=float(sys.argv[3])
print('alpha',alpha)

#ID='671'
#inputname='outfile'+ID+'.npz'
inputname='outfile_crop'+ID+'.npz'
npzfile=np.load(inputname)
datacp=npzfile[npzfile.files[0]] #a copy for original data
#load the ground truth bounding box
bbox= './bbox/'+ID+'_track.txt'
       
#load the image
img= cv2.imread('crop_img'+ID+'.jpg')
#something shift to prevent log on negative number
min_data= np.amin(datacp,axis=2)
min_stack=np.repeat(min_data[:, :, np.newaxis], 12, axis=2)
data=datacp-min_stack+np.finfo(np.float32).eps

	

#segnet classes 0~11
"""
0:Sky
1:Building
2:Pole
3:Road Marking
4:Road
5:Pavement
6:Tree
7:Sign Symbol
8:Fence
9:Vehicle
10:Pedestrian
11:Bike
"""
#define union classes on layer representation
#Sky(L1):0 #Buiding(L2):1,2,6,7,8 #moving object(L3):9,10,11 #road(L4):3,4,5
#start_eng=time.time()
beta=0.1 #param for energy function
#energy function with 6 channels: L1,L2,c9,c10,c11,L4
#pool the min energy in each union
energy=np.zeros((data.shape[0],data.shape[1],6))
datacost=(-1)*beta*np.log(data);
energy[:,:,0]=datacost[:,:,0];
energy[:,:,1]=np.amin(np.dstack((datacost[:,:,1],datacost[:,:,2],datacost[:,:,6],datacost[:,:,7],datacost[:,:,8])),axis=2)
energy[:,:,2:5]=datacost[:,:,9:12];
energy[:,:,5]=np.amin(np.dstack((datacost[:,:,4],datacost[:,:,3],datacost[:,:,5])),axis=2)
end_eng=time.time()

#global IMG_H
#global IMG_W
#global IMG_S
IMG_H=energy.shape[0]
IMG_W=energy.shape[1]
IMG_S=3
###

#start_accu=time.time()
#integral image
Itg=np.cumsum(energy,axis=0).astype(np.float32)
accu0=np.squeeze(Itg[:,:,0])
accu1=np.squeeze(Itg[:,:,1])
accu2=np.squeeze(Itg[:,:,2])
accu3=np.squeeze(Itg[:,:,3])
accu4=np.squeeze(Itg[:,:,4])
accu5=np.squeeze(Itg[:,:,5])
end_accu=time.time()

bigN=np.iinfo('i').max #an infinite float pt num

#Itergral image covert to gpu
accu0_gpu=gpuarray.to_gpu(accu0.astype(np.float32))
accu1_gpu=gpuarray.to_gpu(accu1.astype(np.float32))
accu_L2=np.stack((accu2,accu3,accu4),axis=0)#.reshape(IMG_S*IMG_W*IMG_H)
accu_L2_gpu=gpuarray.to_gpu(accu_L2.astype(np.float32))
accu5_gpu=gpuarray.to_gpu(accu5.astype(np.float32))


kernel_code_template = """
__global__ void MatrixMulKernel(float *a0, float *a1, float *E)
{

    int tx = threadIdx.x;
    int ty = threadIdx.y;
	//E1[:,0,0]=0
	E[tx*%(MATRIX_SIZE)s*%(MATRIX_SIZE)s]=0;	
	//use only one block now
	for(int i=1;i<%(MATRIX_SIZE)s;++i)
		
		for(int j=1;j<i+1;++j)
			E[i* %(MATRIX_SIZE)s+j+tx*%(MATRIX_SIZE)s*%(MATRIX_SIZE)s]=a0[j*%(BLOCK_SIZE)s+tx]+a1[i*%(BLOCK_SIZE)s+tx]-a1[j*%(BLOCK_SIZE)s+tx];

}
__global__ void MatrixMap1d(float *ain, int *hid,float *aout)
{

    int tx = threadIdx.x;
    int ty = threadIdx.y;
	//use only one block now
	aout[tx]=ain[tx*%(MATRIX_SIZE)s +hid[tx] ];

}
__global__ void MatrixMapGuid3(int *ain0, int *ain1, int *ain2, int *hid,int *Guid,int *aout)
{

    int tx = threadIdx.x;
    int ty = threadIdx.y;
	//use only one block now
	aout[tx]=(Guid[tx]==0)*ain0[tx*%(MATRIX_SIZE)s +hid[tx] ];
	aout[tx]+=(Guid[tx]==1)*ain1[tx*%(MATRIX_SIZE)s +hid[tx] ];
	aout[tx]+=(Guid[tx]==2)*ain2[tx*%(MATRIX_SIZE)s +hid[tx] ];

}
__global__ void ArrayMinKernel(float *ain, float *amin, int *argmin)
{

    int tx = threadIdx.x;
    int ty = threadIdx.y;
	int bx=blockIdx.x;
	int by=blockIdx.y;
	int nb=bx*%(MATRIX_SIZE)s;
	
	//use blocks: image_width  threads: image_height
	amin[tx+nb]=ain[tx*%(MATRIX_SIZE)s+nb*%(MATRIX_SIZE)s];
	argmin[tx+nb]=0;
	for(int i=1;i<%(MATRIX_SIZE)s;++i){
		
		float temp=ain[tx*%(MATRIX_SIZE)s+i+nb*%(MATRIX_SIZE)s];
		if(temp<amin[tx+nb]){
			amin[tx+nb]=temp;
			argmin[tx+nb]=i;
		}
	
	}
}
__global__ void ArrayMinKernel3(float *ain, float *amin, int *argmin)
{

    int tx = threadIdx.x;
    int ty = threadIdx.y;
	int bx=blockIdx.x;
	int by=blockIdx.y;
	int nb=bx*%(MATRIX_SIZE)s;
	
	//use blocks: image_width  threads: image_height
	amin[tx+nb]=ain[nb+tx];
	argmin[tx+nb]=0;
	for(int i=1;i<3;++i){
		
		float temp=ain[i*%(BLOCK_SIZE)s*%(MATRIX_SIZE)s+tx+nb];
		if(temp<amin[tx+nb]){
			amin[tx+nb]=temp;
			argmin[tx+nb]=i;
		}
	
	}
				
}		
__global__ void ArgMinKernel(float *ain, int *argmin)
{

    int tx = threadIdx.x;
    int ty = threadIdx.y;
	
	//use only blocks
	float amin=ain[tx*%(MATRIX_SIZE)s];
	argmin[tx]=0;
	for(int i=1;i<%(MATRIX_SIZE)s;++i){
		float temp=ain[tx*%(MATRIX_SIZE)s+i];
		if(temp<amin){
			amin=temp;
			argmin[tx]=i;
		}
	
	}
				
}
__global__ void MatrixFill(float *aE,float *ah)
{
	
        int tx = threadIdx.x;
	int bx =blockIdx.x;
	//note a is an 2d array
	aE[bx*%(MATRIX_SIZE)s+tx]=0;//1000;
	ah[bx*%(MATRIX_SIZE)s+tx]=0;
}
__global__ void MatrixMulKernel2(float *a0, float *a1, float *Q,float *E)
{
	int tx = threadIdx.x;
    int ty = threadIdx.y;
	int bx=blockIdx.x;
	int by=blockIdx.y;
	int nb=bx*%(BLOCK_SIZE)s*%(MATRIX_SIZE)s;
 
	//(h1,h2)=(0,0)
	//E3v[:,0,0]=Itg[energy.shape[0]-1,:,5]+Q[:,0]	
	E[nb*%(MATRIX_SIZE)s+tx*%(MATRIX_SIZE)s*%(MATRIX_SIZE)s]=a1[(%(MATRIX_SIZE)s-1)*%(BLOCK_SIZE)s+tx]+Q[tx*%(MATRIX_SIZE)s];

	//use only one block now
	for(int i=1;i<%(MATRIX_SIZE)s;++i)
		for(int j=0;j<i+1;++j)
			E[i* %(MATRIX_SIZE)s+j+tx*%(MATRIX_SIZE)s*%(MATRIX_SIZE)s+nb*%(MATRIX_SIZE)s]=a0[i*%(BLOCK_SIZE)s+tx+nb]-a0[j*%(BLOCK_SIZE)s+tx+nb]+a1[(%(MATRIX_SIZE)s-1)*%(BLOCK_SIZE)s+tx]-a1[i*%(BLOCK_SIZE)s+tx]+Q[tx*%(MATRIX_SIZE)s+j];

}
__global__ void MatrixMulKernel3(float *a0, float *a1, float *Q,float *cost,float k,float *E)
{
	//float k=1;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
	int bx=blockIdx.x;
	int by=blockIdx.y;
	int nb=bx*%(BLOCK_SIZE)s*%(MATRIX_SIZE)s;
 
	//(h1,h2)=(0,0)
	//E3v[:,0,0]=Itg[energy.shape[0]-1,:,5]+Q[:,0]	
	E[nb*%(MATRIX_SIZE)s+tx*%(MATRIX_SIZE)s*%(MATRIX_SIZE)s]=a1[(%(MATRIX_SIZE)s-1)*%(BLOCK_SIZE)s+tx]+Q[tx*%(MATRIX_SIZE)s];

	//use only one block now
	for(int i=1;i<%(MATRIX_SIZE)s;++i)
		for(int j=0;j<i+1;++j)
			E[i* %(MATRIX_SIZE)s+j+tx*%(MATRIX_SIZE)s*%(MATRIX_SIZE)s+nb*%(MATRIX_SIZE)s]=a0[i*%(BLOCK_SIZE)s+tx+nb]-a0[j*%(BLOCK_SIZE)s+tx+nb]+a1[(%(MATRIX_SIZE)s-1)*%(BLOCK_SIZE)s+tx]-a1[i*%(BLOCK_SIZE)s+tx]+Q[tx*%(MATRIX_SIZE)s+j]+k*cost[i*%(MATRIX_SIZE)s+j+bx*%(MATRIX_SIZE)s*%(MATRIX_SIZE)s];

}
__global__ void MatrixColTalk(float *Q, float *pdist, float *QcE ,int *Qch,float *tempQ,int *h_ref, int *h)
{
    int tx = threadIdx.x;
	int bx=blockIdx.x;
 
	//map2last_E[0,v0:energy.shape[0]-1]=0
	QcE[bx]=0;
	int lines=0;
	for (int c=1;c<%(BLOCK_SIZE)s;++c){
		if(h[c]!=h_ref[c]){
			if(h[c-1]!=h_ref[c-1]){
			for (int k=0; k<%(MATRIX_SIZE)s; ++k)
				tempQ[bx*%(MATRIX_SIZE)s+k]=Q[c*%(MATRIX_SIZE)s+bx]+Q[(c-1)*%(MATRIX_SIZE)s+k]+QcE[(c-1)*%(MATRIX_SIZE)s+k]+pdist[bx*%(MATRIX_SIZE)s+k];
			//minimize
			for (int i=0; i<%(MATRIX_SIZE)s; ++i)
				if(tempQ[bx*%(MATRIX_SIZE)s+i]<QcE[c*%(MATRIX_SIZE)s+bx]){
					QcE[c*%(MATRIX_SIZE)s+bx]=tempQ[bx*%(MATRIX_SIZE)s+i];
					Qch[c*%(MATRIX_SIZE)s+bx]=i;
				}
			lines++;
			} else{
			float min_value=QcE[(c-1)*%(MATRIX_SIZE)s];
			for (int i=1; i<%(MATRIX_SIZE)s; ++i){
				float temp=QcE[(c-1)*%(MATRIX_SIZE)s+i];
				if(temp<min_value)
					h[c-1]=i;
			}	
			//back infer	
			for (int i=c-1; i>c-lines; --i)
				h[i-1]=Qch[i*%(MATRIX_SIZE)s+h[i]];
			QcE[(c-1)*%(MATRIX_SIZE)s+bx]=0;
			lines=0;
			}
		}else{
			lines=0;		
			QcE[(c)*%(MATRIX_SIZE)s+bx]=0;
		}
	}
	//h1[energy.shape[1]-1]=np.argmin(map2last_E[energy.shape[1]-1,:])#head to back 
	//for i in range(energy.shape[1]-1,energy.shape[1]-lines,-1):
        	//h1[i-1]=map2last_h[i,h1[i]]
	//last column	
	float min_value=QcE[(%(BLOCK_SIZE)s-1)*%(MATRIX_SIZE)s];
	for (int i=1; i<%(MATRIX_SIZE)s; ++i){
		float temp=QcE[(%(BLOCK_SIZE)s-1)*%(MATRIX_SIZE)s+i];
		if(temp<min_value)
			h[%(BLOCK_SIZE)s-1]=i;
	}
	//back infer	
	for (int i=%(BLOCK_SIZE)s-1; i>%(BLOCK_SIZE)s-lines; --i)
		h[i-1]=Qch[i*%(MATRIX_SIZE)s+h[i]];

}
"""
kernel_code = kernel_code_template % {
	'MATRIX_SIZE': IMG_H, 
	'BLOCK_SIZE': IMG_W 
}

# compile the kernel code 
mod = compiler.SourceModule(kernel_code)

#start_func=time.time()
# get the kernel function from the compiled module
matrixmul = mod.get_function("MatrixMulKernel")
arraymin = mod.get_function("ArrayMinKernel")
arraymin3 = mod.get_function("ArrayMinKernel3")
argmin1d = mod.get_function("ArgMinKernel")
matrixmap1d = mod.get_function("MatrixMap1d")
matrixmapguid3 = mod.get_function("MatrixMapGuid3")
matrixmul2 = mod.get_function("MatrixMulKernel2")
matrixmul3 = mod.get_function("MatrixMulKernel3")
matrixfill = mod.get_function("MatrixFill")
matrixcoltalk = mod.get_function("MatrixColTalk")

end_preprocessing= time.time()

def visualize( data, fold, ID, para):

	#input_seg='outfile_crop'+ID+'.npz'	#Segnet
	#npzfile=np.load(input_seg)
	#npzfile3=np.load(input_seg_prob1)
	#data=npzfile['data']

	#show segmentation
	label_colours = cv2.imread('camvid12.png').astype(np.uint8)
	catg= data.astype(np.uint8)
	segmentation_ind_3ch = np.resize(catg,(3,catg.shape[0],catg.shape[1]))
	segmentation_ind_3ch = segmentation_ind_3ch.transpose(1,2,0).astype(np.uint8)
	segmentation_rgb = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)
	cv2.LUT(segmentation_ind_3ch,label_colours,segmentation_rgb)

	segmentation_rgb_2= copy.copy(segmentation_rgb)
	#draw horizon line
#	cv2.line(segmentation_rgb,(0,horizon),(segmentation_rgb.shape[1]-1,horizon),(255,255,255),2)

	
	ori_img = cv2.imread('re_img'+ID+'.jpg')
	
	combine_img= segmentation_rgb_2*0.6+ori_img*0.4  #horizon+layer+talk

	#	480*640 image
	cv2.imwrite( '/home/nmsoc/code/SegNet/progress/cityscapesScripts/results_small'+str(fold)+'/rgb/'+ID+para+'.png', segmentation_rgb_2)

	cv2.imwrite( '/home/nmsoc/code/SegNet/progress/cityscapesScripts/results_small'+str(fold)+'/combine/'+ID+para+'.png', combine_img);

	#	1024*2048 image
	segmentation_rgb_2= cv2.resize(segmentation_rgb_2,(2048,1024),interpolation=cv2.INTER_NEAREST)
	cv2.imwrite( '/home/nmsoc/code/SegNet/progress/cityscapesScripts/results'+str(fold)+'/rgb/'+ID+para+'.png', segmentation_rgb_2)

	combine_img= cv2.resize(combine_img,(2048,1024),interpolation=cv2.INTER_NEAREST)
	cv2.imwrite( '/home/nmsoc/code/SegNet/progress/cityscapesScripts/results'+str(fold)+'/combine/'+ID+para+'.png', combine_img);

	labelID= cv2.resize(catg,(2048,1024),interpolation=cv2.INTER_NEAREST)
	cv2.imwrite('/home/nmsoc/code/SegNet/progress/cityscapesScripts/results'+str(fold)+'/labelID/'+ID+para+'.png',labelID)

def PixelClass(L2, h1, h2, h3):

	#following is code for saving layer label for visaulization
	smallTable1=[1,2,6,7,8]
	smallTable2=[4,4,5]
	disp_seg=np.zeros((data.shape[0],data.shape[1]))

	for j in range(data.shape[1]):
		disp_seg[:h3[j],j]=0;

		for i in range(h3[j],h2[j]):
			disp_seg[i,j]=smallTable1[np.argmax([datacp[i,j,1],datacp[i,j,2],datacp[i,j,6],datacp[i,j,7],datacp[i,j,8]],axis=0)]             
		if(L2[j]==0): 
			disp_seg[h2[j]:h1[j],j]=9
		elif(L2[j]==1):
			disp_seg[h2[j]:h1[j],j]=10
		elif(L2[j]==2):
			disp_seg[h2[j]:h1[j],j]=11 
		for i in range(h1[j],data.shape[0]):
			disp_seg[i,j]=smallTable2[np.argmax(datacp[i,j,3:6],axis=0)]
	
	return disp_seg
	

def infering(Q_h3_gpu, E3_gpu):

	E3v_gpu=E3_gpu[:IMG_W*IMG_H*IMG_H]
	E3p_gpu=E3_gpu[IMG_W*IMG_H*IMG_H:2*IMG_W*IMG_H*IMG_H]
	E3b_gpu=E3_gpu[IMG_W*IMG_H*IMG_H*2:IMG_W*IMG_H*IMG_H*3]
	 
	Qv_gpu = gpuarray.empty((IMG_W,IMG_H), np.float32)
	Qv_h2_gpu = gpuarray.empty((IMG_W,IMG_H), np.int32)
	Qp_gpu = gpuarray.empty((IMG_W,IMG_H), np.float32)
	Qp_h2_gpu = gpuarray.empty((IMG_W,IMG_H), np.int32)
	Qb_gpu = gpuarray.empty((IMG_W,IMG_H), np.float32)
	Qb_h2_gpu = gpuarray.empty((IMG_W,IMG_H), np.int32)

	#print '%30s' % 'misc execution time ', str((end_misc - start_misc)*1000), 'ms'
	#arraymin_v = mod0.get_function("ArrayMinKernel")
	#start_min2=time.time()
	arraymin(
	    	# inputs
	    	E3v_gpu, 
	    	# output
	    	Qv_gpu,Qv_h2_gpu,#d_gpu, 
	    	# (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
	     	block = (IMG_H, 1, 1), #modify later
			grid=(IMG_W,1,1),
	    )

	#arraymin_p = mod0.get_function("ArrayMinKernel")
	arraymin(
	    	# inputs
	    	E3p_gpu, 
	    	# output
	    	Qp_gpu,Qp_h2_gpu,#d_gpu, 
	    	# (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
	     	block = (IMG_H, 1, 1), #modify later
			grid=(IMG_W,1,1),
	    )

	#arraymin_b = mod0.get_function("ArrayMinKernel")
	arraymin(
	    	# inputs
	    	E3b_gpu, 
	    	# output
	    	Qb_gpu,Qb_h2_gpu,#d_gpu, 
	    	# (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
	     	block = (IMG_H, 1, 1), #modify later
			grid=(IMG_W,1,1),
	    )
	#end_min2=time.time()
	#print '%30s' % 'min+argmin x3  execution time ', str((end_min2 - start_min2)*1000), 'ms'


	#start_move=time.time()
	Q_stack_gpu=gpuarray.empty((3,IMG_W,IMG_H),np.float32)
	Q_stack_gpu[0]=Qv_gpu#.reshape(IMG_W*IMG_H)
	Q_stack_gpu[1]=Qp_gpu#.reshape(IMG_W*IMG_H)
	Q_stack_gpu[2]=Qb_gpu#.reshape(IMG_W*IMG_H)
	#end_move=time.time()
	#print '%30s' % 'move data in gpu time ', str((end_move - start_move)*1000), 'ms'

	#=([Qv_gpu,Qp_gpu,Qb_gpu])
	##print('stack shape',Q_stack_gpu.shape)
	##print('stack shape',Q_stack_gpu[0].shape)
	# call the kernel on the card
	#start_empt=time.time()
	Q2_gpu=gpuarray.empty((IMG_W,IMG_H),np.float32)
	Q2c_gpu=gpuarray.empty((IMG_W,IMG_H),np.int32)

	h1_gpu=gpuarray.empty((IMG_W),np.int32)
	L2_gpu=gpuarray.empty((IMG_W),np.int32)
	h2_gpu=gpuarray.empty((IMG_W),np.int32)
	h3_gpu=gpuarray.empty((IMG_W),np.int32)
	#end_empt=time.time()
	#print '%30s' % 'create empty a in gpu time ', str((end_empt - start_empt)*1000), 'ms'

	#start_min3=time.time()
	arraymin3(
	    	# inputs
	    	Q_stack_gpu, 
	    	# output
	    	Q2_gpu,Q2c_gpu,#d_gpu, 
	    	# (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
	     	block = (IMG_H, 1, 1), #modify later
			grid=(IMG_W,1,1),
	    )

	#end_min3=time.time()
	#print '%30s' % 'min v,p,b excution time ', str((end_min3 - start_min3)*1000), 'ms'

	#start_get=time.time()
	#Q2=Q2_gpu.get()
	#Q2c=Q2c_gpu.get()

	#end_get=time.time()
	#print '%30s' % 'get Q2 from gpu time ', str((end_get - start_get)*1000), 'ms'
	#start_mins=time.time()
	argmin1d(
	    	# inputs
	    	Q2_gpu, 
	    	# output
	    	h1_gpu,#d_gpu, 
	    	# (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
	     	block = (IMG_W, 1, 1), #modify later
			grid=(1,1,1),
	    )


	matrixmap1d(
	    	# inputs
	    	Q2c_gpu, h1_gpu,
	    	# output
	    	L2_gpu,#d_gpu, 
	    	# (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
	     	block = (IMG_W, 1, 1), #modify later
			grid=(1,1,1),
	    )
	#end_mins=time.time()
	#print '%30s' % 'minimize somthing execution time ', str((end_mins - start_mins)*1000), 'ms'

	#start_guid=time.time()
	matrixmapguid3(
	    	# inputs
	    	Qv_h2_gpu, #h1_gpu,#L2_gpu,#(L2_gpu==0),
	    	Qp_h2_gpu, #hp_gpu,#L2_gpu,#(L2_gpu==0),
	    	Qb_h2_gpu, h1_gpu,L2_gpu,#(L2_gpu==0),
	    	# output
	    	h2_gpu,#d_gpu, 
	    	# (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
	     	block = (IMG_W, 1, 1), #modify later
			grid=(1,1,1),
	    )
	L2=L2_gpu.get()
	h2=h2_gpu.get()

	#end_guid=time.time()
	#print '%30s' % 'guid min exec time ', str((end_guid - start_guid)*1000), 'ms'

	#h2_gpu=(L2_gpu==0)*h2_v_gpu+(L2==1)*h2_p_gpu+(L2==2)*h2_b_gpu

	#h3_gpu=gpuarray.empty((IMG_W),np.int32)
	#start_map=time.time()
	matrixmap1d(
	    	# inputs
	    	Q_h3_gpu, h2_gpu,
	    	# output
	    	h3_gpu,#d_gpu, 
	    	# (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
	     	block = (IMG_W, 1, 1), #modify later
			grid=(1,1,1),
	    )
	#end_map=time.time()
	#print '%30s' % 'map h in gpu time ', str((end_map - start_map)*1000), 'ms'

	#start_geth=time.time()
	h1=h1_gpu.get()
	h3=h3_gpu.get()
	#end_geth=time.time()
	#print '%30s' % 'get h(w) from gpu time ', str((end_geth - start_geth)*1000), 'ms'

	disp_seg= PixelClass(L2, h1, h2, h3)

	return disp_seg, L2, h1_gpu, h2_gpu, h3_gpu, Q2_gpu

def layer(Q_gpu,Q_h3_gpu):
	#minimize total energy
	#seperate to car,people,bike 3 shannels

	#start_empA=time.time()

	E3_gpu=gpuarray.empty((IMG_S*IMG_W*IMG_H*IMG_H),np.float32).fill(100.0)#+1.0
	#end_empA=time.time()
	#print '%30s' % 'create number filled gpu time ', str((end_empA - start_empA)*1000), 'ms'


	###
	#start_gpu2=time.time()
	# call the kernel on the card
	matrixmul2(
	    	# inputs
	    	accu_L2_gpu, accu5_gpu, Q_gpu, 

	    	# output
	    	E3_gpu,
	    	# (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
	     	block = (IMG_W, 1, 1), #modify later
			grid=(IMG_S,1,1),	
	    )
	
	#end_gpu2=time.time()
	#print '%30s' % 'gpu2 execution time ', str((end_gpu2 - start_gpu2)*1000), 'ms' #str(gpuStart.time_till(gpuEnd)*(1000)),'ms' #str((end_gpu - start_gpu)*1000), 'ms'
	disp_seg, L2, h1_gpu, h2_gpu, h3_gpu, Q2_gpu= infering(Q_h3_gpu, E3_gpu)
	np.savez('/home/nmsoc/code/SegNet/progress/cityscapesScripts/results_small/layer/seg_layer_'+ID,data=disp_seg)
	visualize( disp_seg, '/layer/', ID, '')

def layer_horizontal(Q_gpu,Q_h3_gpu):
	
	#start_geocost=time.time()
	# add the horizon contraint
	v0= horizon(bbox,img)
	print 'horizon line:', v0
	yv_c= 1.50  	#vehicle 3D height (m)
	yv_b= 2.10  	#vehicle 3D height (m)
	yp= 1.60  	#person 3D height (m)
	yb= 1.20  	#bike 3D height (m)
	yc= 1.40	#camera height
	#v0= 120	#horizon line #240  1#200   2#120   5#160  8#250   9#235
	#start_geocost=time.time()

	vi= np.arange(0,data.shape[0])          #object position on image 

	hv_c= yv_c*abs(vi-v0)/yc
	hv_b= yv_b*abs(vi-v0)/yc
	hp= yp*abs(vi-v0)/yc
	hb= yb*abs(vi-v0)/yc
	hv_c[:v0]=np.iinfo('i').max 		 
	hv_b[:v0]=np.iinfo('i').max
	hp[:v0]=np.iinfo('i').max
	hb[:v0]=np.iinfo('i').max

	#create table for geometric cost for object depth Qh(h1,h2,class)
	#seperate 3 classes: vihecle, poeple, bike
	v_c_v=np.array(range(energy.shape[0]))	#all possible h1
	v_b_v=np.array(range(energy.shape[0])) 
	p_v=np.array(range(energy.shape[0]))
	b_v=np.array(range(energy.shape[0]))

	best_v_c_u=v_c_v-hv_c	#hope ideal h2 for every h1
	best_v_b_u=v_b_v-hv_b
	best_p_u=p_v-hp
	best_b_u=b_v-hb

	best_v_c_u=np.repeat(best_v_c_u[:,np.newaxis], energy.shape[0], axis=1)	#extend 
	best_v_b_u=np.repeat(best_v_b_u[:,np.newaxis], energy.shape[0], axis=1)
	best_p_u=np.repeat(best_p_u[:,np.newaxis], energy.shape[0], axis=1)
	best_b_u=np.repeat(best_b_u[:,np.newaxis], energy.shape[0], axis=1)

	hu=np.array(range(energy.shape[0])) #all possible h2 according to every h1 
	hu=np.repeat(hu[np.newaxis,:], energy.shape[0], axis=0)

	geo_cost_v_c=abs(best_v_c_u-hu)**2	#define cost by distance(real h2,ideal h2)
	geo_cost_v_b=abs(best_v_b_u-hu)**2
	geo_cost_p=abs(best_p_u-hu)**2
	geo_cost_b=abs(best_b_u-hu)**2

	np.fill_diagonal(geo_cost_v_c, 0) #if h1==h2: cost=0
	np.fill_diagonal(geo_cost_v_b, 0) 
	np.fill_diagonal(geo_cost_b, 0)
	np.fill_diagonal(geo_cost_p, 0)


	geo_cost_v=np.amin(np.dstack((geo_cost_v_c,geo_cost_v_b)),axis=2)
	#end_geocost=time.time()
	#print '%30s' % 'geo_cost  execution time ', str((end_geocost - start_geocost)*1000), 'ms'
	###
	geo_cost=np.stack((geo_cost_v,geo_cost_p,geo_cost_b),axis=0)
	geo_cost_gpu=gpuarray.to_gpu(geo_cost.astype(np.float32))

	#minimize total energy
	#seperate to car,people,bike 3 shannels

	#start_empA=time.time()

	E3_gpu=gpuarray.empty((IMG_S*IMG_W*IMG_H*IMG_H),np.float32).fill(100.0)#+1.0
	#end_empA=time.time()
	#print '%30s' % 'create number filled gpu time ', str((end_empA - start_empA)*1000), 'ms'


	###
	#start_gpu2=time.time()
	# call the kernel on the card
	#k=1

	matrixmul3(
	    	# inputs
	    	accu_L2_gpu, accu5_gpu, Q_gpu,geo_cost_gpu, 
	    	#weight
	    	np.float32(k),
	    	# output
	    	E3_gpu,
	    	
	    	# (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
	     	block = (IMG_W, 1, 1), #modify later
			grid=(IMG_S,1,1),

	    )
	#end_gpu2=time.time()
	#print '%30s' % 'gpu2 execution time ', str((end_gpu2 - start_gpu2)*1000), 'ms' #str(gpuStart.time_till(gpuEnd)*(1000)),'ms' #str((end_gpu - start_gpu)*1000), 'ms'
	disp_seg, L2, h1_gpu, h2_gpu, h3_gpu, Q2_gpu= infering(Q_h3_gpu, E3_gpu)
	np.savez('/home/nmsoc/code/SegNet/progress/cityscapesScripts/results_small/horizontal/seg_horizon_'+ID,para=k,hori=v0,data=disp_seg)	# layer+horizontal
	para= '_k='+str(k)
	visualize( disp_seg, '/horizontal/', ID, para)
	return Q2_gpu, L2, h1_gpu, h2_gpu, h3_gpu, v0

def layer_horizontal_talk( Q2_gpu, L2, h1_gpu, h2_gpu, h3_gpu, v0):

	map2pre_E_gpu=gpuarray.empty((energy.shape[1],energy.shape[0]),np.float32)
	map2pre_h_gpu=gpuarray.empty((energy.shape[1],energy.shape[0]),np.int32)

	tempQ_gpu=gpuarray.empty((energy.shape[0],energy.shape[0]),np.float32)

	di=np.tile(np.arange(energy.shape[0]),(energy.shape[0],1))
	dj=di.T#np.transpose(di)
	dij=abs(di-dj)*alpha
	dij_gpu=gpuarray.to_gpu(dij.astype(np.float32))
	#print(dij_gpu)
	"""
	matrixfill(
	    	# inputs
	    	map2pre_E_gpu, map2pre_h_gpu,
	    	# (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
	     	block = (IMG_H, 1, 1), #modify later
		grid=(IMG_W,1,1),
	    )
	matrixcoltalk(
	    	# inputs
	    	Q2_gpu, dij_gpu,map2pre_E_gpu,map2pre_h_gpu,tempQ_gpu,h2_gpu,
	    	# output
	    	h1_gpu,#d_gpu, 
	    	# (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
	     	block = (1, 1, 1), #modify later
		grid=(IMG_H,1,1),
	    )
	"""
	h1=h1_gpu.get()
	h2=h2_gpu.get()
	h3=h3_gpu.get()

	#print(h1)
	#columes talk
	#start_talk=time.time()i
	map2last_E=1000*np.ones((energy.shape[1],energy.shape[0]))
	map2last_h=np.zeros((energy.shape[1],energy.shape[0]))

	map2last_E[0,v0:energy.shape[0]-1]=0

	lines=0
	Q2=Q2_gpu.get()
	for c in range(1,energy.shape[1]):
		
		if(h1[c-1]!=h2[c-1]):
			if(h1[c]!=h2[c]):        
				#cross_cost=np.transpose(npmat.repmat(Q2[c,:],energy.shape[0],1))+npmat.repmat(Q2[c-1,:],energy.shape[0],1)
				#cross_cost=cross_cost+npmat.repmat(map2last_E[c-1,:],energy.shape[0],1)+alphi*dij
				cross_cost=(np.tile(Q2[c,:],(energy.shape[0],1))).T+np.tile(Q2[c-1,:],(energy.shape[0],1))
				#cross_cost=(np.tile(Q2[c,:],(energy.shape[0],1))).T
				cross_cost=cross_cost+np.tile(map2last_E[c-1,:],(energy.shape[0],1))+dij
	                	map2last_E[c,:]=np.amin(cross_cost,axis=1)
	                	map2last_h[c,:]=np.argmin(cross_cost,axis=1)
				lines+=1
			else:
				#map2last_E[c-1:c+1,:]=0
				h1[c-1]=np.argmin(map2last_E[c-1,:])#head to back 
				for i in range(c-1,c-lines,-1):
					#if(map2last_h[i,h1[i]]):
					h1[i-1]=map2last_h[i,h1[i]]
				map2last_E[c-1,:]=0
				lines=0
		else:
			lines=0	
	h1[energy.shape[1]-1]=np.argmin(map2last_E[energy.shape[1]-1,:])#head to back 
	for i in range(energy.shape[1]-1,energy.shape[1]-lines,-1):
	        h1[i-1]=map2last_h[i,h1[i]]

	h1=(h1==0)*v0+h1

	disp_seg= PixelClass(L2, h1, h2, h3)
	np.savez('/home/nmsoc/code/SegNet/progress/cityscapesScripts/results_small/talk/seg_talk_'+ID,para=k,data=disp_seg,alpha=alpha)
	para= '_k='+str(k)+'_alpha='+str(alpha)
	visualize( disp_seg, '/talk/', ID, para)

if __name__ == '__main__':

	#SegNet
	catg=np.argmax(datacp,axis=2).astype(np.uint8)
	visualize( catg, '/SegNet/', ID,'')
	
	start_layer= time.time()
	E1_gpu=gpuarray.empty((energy.shape[1]*energy.shape[0]*energy.shape[0]),np.float32).fill(100.0)

	# call the kernel on the card
	matrixmul(
	    	# inputs
	    	accu0_gpu, accu1_gpu, 
	    	# output
	    	E1_gpu,#d_gpu, 
	    	# (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
	     	block = (IMG_W, 1, 1), #modify later
			grid=(1,1,1),
	    )
	#gpuEnd.record()
	#end_gpu=time.time()#gpuEnd.synchronize()#time.time()
	#print '%30s' % 'gpu1 execution time ', str((end_gpu - start_gpu)*1000), 'ms' #str(gpuStart.time_till(gpuEnd)*(1000)),'ms' #str((end_gpu - start_gpu)*1000), 'ms'
	Q_gpu = gpuarray.empty((IMG_W,IMG_H), np.float32)
	Q_h3_gpu = gpuarray.empty((IMG_W,IMG_H), np.int32)

	#start_min=time.time()                
	# call the kernel on the card
	arraymin(
	    	# inputs
	    	E1_gpu, 
	    	# output
	    	Q_gpu,Q_h3_gpu,#d_gpu, 
	    	# (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
	     	block = (IMG_H, 1, 1), #modify later
			grid=(IMG_W,1,1),
	    )
	
	#end_min=time.time()
	#print '%30s' % 'min+argmin execution time ', str((end_min - start_min)*1000), 'ms'
	
	#processing layer

	mid_layer= time.time()
	layer(Q_gpu,Q_h3_gpu)
	end_layer= time.time()
	
	#processing layer_horizontal
	Q2_gpu, L2, h1_gpu, h2_gpu, h3_gpu, v0= layer_horizontal(Q_gpu,Q_h3_gpu)
	end_hori= time.time()

	#processing layer_horizontal_talk
	layer_horizontal_talk(Q2_gpu, L2,h1_gpu, h2_gpu, h3_gpu, v0)
	end_talk= time.time()

	print '%40s' % 'preproccessing time: ',str((end_preprocessing - start)*1000), 'ms'
	print '%40s' % 'layer execution time: ',str((end_layer - end_preprocessing)*1000), 'ms'
	print '%40s' % 'layer+horizontal execution time: ',str((end_hori - end_layer + mid_layer - start_layer)*1000), 'ms'
	print '%40s' % 'layer+horizontal+talk execution time: ',str((end_talk - end_layer + mid_layer - start_layer)*1000), 'ms'
	print '%40s' % 'total time: ',str((end_talk - start)*1000), 'ms'
	print '%40s' % '------------------------finish--------------------------'
