import numpy as np
import pdb

def horizon_calculated(gt, im_count, camera_height = 1.3):
    """ 
    'gt' is ground_truth 
    input 'gt': the whole ground truth data. The method that how to load the ground truth
                    is in the 'if __name__ == "__main__"'
    
    output 'horizon_lines': all horizon lines in each frame.
    """

    HEIGHT = {'bicycle':1.05, 
              'bus':2.60, 
              'car':1.50, 
              'motorbike':1.30,
              'person':1.70
             }
    
    frame_ind = im_count+1
    
    while(frame_ind <= (im_count + 100)):
        horizon_line = 0
        bbox_len = 0
        for cls in gt.keys():
            if not np.count_nonzero(gt[cls]):
                continue
            valid_ind = (gt[cls][:,0] == frame_ind)
            gt_bbox = gt[cls][valid_ind, 1:]
            if len(gt_bbox) == 0:
                continue
            
            for ind in range(0, len(gt_bbox)):
                # ground truth boxes in one frame
                bbox = gt_bbox[ind,:]
                
                # sum up all the horizon line calculated by each boxes in one frame
                # the algorithm is according to the paper (Putting Objects in Perspective)
                horizon_line = horizon_line + \
                               int(bbox[3]) - ((int(bbox[3])-int(bbox[1]))*camera_height/HEIGHT.setdefault(cls, 1.5))
            
            bbox_len = bbox_len + len(gt_bbox)

        # average horizon line in one frame
        if bbox_len != 0:
            try:
                horizon_lines = np.vstack((horizon_lines, horizon_line/bbox_len))
            except:
                horizon_lines = horizon_line/bbox_len
        else:
            try:
                horizon_lines = np.vstack((horizon_lines, 0))
            except:
                horizon_lines = 0
           
        frame_ind = frame_ind + 1
    return horizon_lines
    
