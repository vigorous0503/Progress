import numpy as np
import cv2
import pdb
import copy
import Image, ImageDraw
import sys
#import matplotlib.pyplot as plt
#ID='780'
ID= sys.argv[1]
'''
input_seg='outfile'+ID+'.npz'
#input_seg_prob='layers_v5_'+ID+'.npz'
input_seg_prob='seg00'+ID+'.npz'
'''

#input_seg='outfile_crop'+ID+'.npz'
#input_seg_prob='seg_crop'+ID+'.npz'


input_seg='outfile_crop'+ID+'.npz'	#Segnet
#input_seg_prob='seg_crop_horizon'+ID+'.npz'	#layer+horizon
input_seg_prob='seg_crop_horizon_talk'+ID+'.npz'	#layer+horizon+talk
input_seg_prob1='seg_crop'+ID+'.npz'	#layer

#load segnet data
#npzfile2.keys()
npzfile=np.load(input_seg)
npzfile2=np.load(input_seg_prob)
npzfile3=np.load(input_seg_prob1)
data=npzfile[npzfile.files[0]]
data2=npzfile2['data']
data3=npzfile3[npzfile3.files[0]]
horizon= npzfile2['hori']
k= npzfile2['para']
#degree= npzfile2['degree']
alpha= npzfile2['alpha']

#show segmentation
label_colours = cv2.imread('camvid12.png').astype(np.uint8)
pdb.set_trace()
catg=np.argmax(data,axis=2).astype(np.uint8)  #Segnet
segmentation_ind_3ch = np.resize(catg,(3,catg.shape[0],catg.shape[1]))
segmentation_ind_3ch = segmentation_ind_3ch.transpose(1,2,0).astype(np.uint8)
segmentation_rgb = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)
cv2.LUT(segmentation_ind_3ch,label_colours,segmentation_rgb)

catg2=data2.astype(np.uint8)
layer_ind_3ch = np.resize(catg2,(3,catg.shape[0],catg.shape[1]))
layer_ind_3ch = layer_ind_3ch.transpose(1,2,0).astype(np.uint8)
layer_rgb = np.zeros(layer_ind_3ch.shape, dtype=np.uint8)
cv2.LUT(layer_ind_3ch,label_colours,layer_rgb)

catg3=data3.astype(np.uint8)
layer_ind_3ch_2 = np.resize(catg3,(3,catg.shape[0],catg.shape[1]))
layer_ind_3ch_2 = layer_ind_3ch_2.transpose(1,2,0).astype(np.uint8)
layer_rgb2 = np.zeros(layer_ind_3ch_2.shape, dtype=np.uint8)
cv2.LUT(layer_ind_3ch_2,label_colours,layer_rgb2)
#display = cv2.applyColorMap((catg**4)*255, cv2.COLORMAP_JET)
#display2 = cv2.applyColorMap(catg2, cv2.COLORMAP_JET)

'''
cv2.imshow("segment",segmentation_rgb)
cv2.imshow("layers",layer_rgb)
cv2.imwrite( './results/segment'+ID+'.jpg', segmentation_rgb );
#cv2.imwrite( './results/layers_v5_'+ID+'.jpg', layer_rgb );
cv2.imwrite( './results/seg00'+ID+'.jpg', layer_rgb );
#cv2.waitKey(0)
'''
'''
#cv2.imshow("segment_crop",segmentation_rgb)
#cv2.imshow("layers_crop",layer_rgb)
cv2.imwrite( './results/segment_crop'+ID+'.jpg', segmentation_rgb );
#cv2.imwrite( './results/layers_v5_'+ID+'.jpg', layer_rgb );
cv2.imwrite( './results/seg_crop'+ID+'.jpg', layer_rgb );
#cv2.waitKey(0)
#cv2.destroyAllWindows()
'''
#seg_layer_hori= copy.copy(layer_rgb)
seg_layer_hori_talk= copy.copy(layer_rgb)
seg_layer= copy.copy(layer_rgb2)
segnet= copy.copy(segmentation_rgb)
#draw horizon line
cv2.line(layer_rgb,(0,horizon),(layer_rgb.shape[1]-1,horizon),(255,255,255),2)
cv2.line(layer_rgb2,(0,horizon),(layer_rgb.shape[1]-1,horizon),(255,255,255),2)



#visualize
crop_img = cv2.imread('crop_img'+ID+'.jpg')

frame = cv2.resize(crop_img, (480,360))
cv2.imwrite('re_img'+ID+'.jpg', frame)
temp1= frame	#draw hoirizon line on original(360*480) image
#temp1 = cv2.imread('./results/ori_crop'+ID+'.jpg')
cv2.line(temp1,(0,horizon),(temp1.shape[1]-1,horizon),(0,0,255),2)
cv2.imwrite( './results/ori_crop'+ID+'.jpg',temp1 );

ori_img = cv2.imread('re_img'+ID+'.jpg')
#combine_hori= seg_layer_hori*0.6+ori_img*0.4  #horizon+layer
combine_hori_talk= seg_layer_hori_talk*0.6+ori_img*0.4  #horizon+layer+talk
combine_layer= seg_layer*0.6+ori_img*0.4  #layer 
combine_segnet= segnet*0.6+ori_img*0.4	#Segnet
#cv2.line(combine_hori,(0,horizon),(combine_hori.shape[1]-1,horizon),(255,255,255),2)
cv2.line(combine_hori_talk,(0,horizon),(combine_hori_talk.shape[1]-1,horizon),(255,255,255),2)
cv2.line(combine_layer,(0,horizon),(combine_layer.shape[1]-1,horizon),(255,255,255),2)
cv2.line(combine_segnet,(0,horizon),(combine_segnet.shape[1]-1,horizon),(255,255,255),2)
cv2.imwrite( './results/segment_crop'+ID+'.jpg', segmentation_rgb );
cv2.imwrite( './results/seg_crop'+ID+'.jpg', layer_rgb2 );
#cv2.imwrite('./results/pycuda_combine_horizon'+ID+'_'+str(k)+'.jpg',combine_hori.astype(np.uint8))
#cv2.imwrite('./results/combine_talk'+ID+'_'+str(k)+'_alpha'+str(alpha)+'_degree'+str(degree)+'.jpg', combine_hori_talk.astype(np.uint8))
#cv2.imwrite('./results/pycuda_v8_combine_talk'+ID+'_'+str(k)+'_alpha'+str(alpha)+'_degree'+str(degree)+'.jpg', combine_hori_talk.astype(np.uint8))
cv2.imwrite('./results/pycuda_v8_combine_talk'+ID+'_'+str(k)+'_alpha'+str(alpha)+'.jpg', combine_hori_talk.astype(np.uint8))
cv2.imwrite('./results/combine_layer'+ID+'.jpg',combine_layer.astype(np.uint8))
cv2.imwrite('./results/combine_SegNet'+ID+'.jpg',combine_segnet)
#cv2.imwrite( './results/combine_talk'+ID+'_'+str(k)+'_alpha'+str(alpha)+'_degree'+str(degree)+'.jpg', combine_img.astype(np.uint8))

cv2.waitKey(0)


'''
#draw ideal_H on layer_image


draw = ImageDraw.Draw(layer_rgb2)
draw.point((np.arrange(0,480),ideal_H),fill = (0,0,0))
del draw

# write to stdout
im.save(sys.stdout, "point.jpg")
'''
