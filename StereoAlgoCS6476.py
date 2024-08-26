import cv2
import numpy as np
from matplotlib import cm
import os
import numpy as np
from numpy.linalg import norm
import cv2
from scipy import ndimage
import time

def pair2grey(left, right, nptype=np.float32):
    left_grey = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY).astype(nptype)
    right_grey = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY).astype(nptype)
    return left_grey, right_grey

# load the images from dataset-2005 subdirectories
def load_image(path, left_suffix='0.png', right_suffix='1.png', scale = 0.5):
    # load images
    left_images = {}
    right_images = {}
    for folder in os.listdir(path):
        # print(folder)
        for file in os.listdir(path + folder + '/'):
            if file.endswith(left_suffix):
                left_img = cv2.imread(path + folder+ '/' + file)
                left_img = cv2.resize(left_img, (0, 0), fx=scale, fy=scale)
                left_images[folder] = left_img
            elif file.endswith(right_suffix):
                right_img = cv2.imread(path + folder + '/' + file)
                right_img = cv2.resize(right_img, (0, 0), fx=scale, fy=scale)
                right_images[folder] = right_img
    return left_images, right_images


def view_images(left_images, right_images):
    for k in left_images.keys():
        cv2.imshow('left', left_images[k])
        cv2.imshow('right', right_images[k])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def disp2grey(disp):
    image = np.zeros((disp.shape[0], disp.shape[1], 3), dtype=np.uint8)
    occluded = (disp < 0)
    image[:] = np.where(occluded, 0, 255 * disp / disp.max())[:, :, np.newaxis]
    image[occluded] = [0, 0, 0]
    return image

def disp2jet(disp):
    cm_jet = cm.ScalarMappable(cmap='jet')
    occluded = (disp < 0)
    jet = cm_jet.to_rgba(np.where(occluded, 0, disp), bytes=True)[:, :, :3]
    jet[occluded] = 0
    return jet

def get_accuracy(GroundTruth, disparity, scale):
    is_correct = np.abs(GroundTruth - disparity) <= 2. * scale
    total_accuracy = is_correct.sum() / float(GroundTruth.size)
    is_visible = GroundTruth >= 0
    visible_accuracy = is_correct[is_visible].sum() / float(is_visible.sum())
    return total_accuracy, visible_accuracy

def ssd_(left, right, window_size=5, search_depth=30):
    left, right = pair2grey(left, right)
    kernel = np.ones((window_size, window_size), np.float32)

    min_ssd = np.full(left.shape, float('inf'), dtype=np.float32)
    labels = np.zeros(left.shape, dtype=np.int)
    for i in range(search_depth):
        shift_right = right if i == 0 else right[:, :-i]
        raw_ssd = np.square(left[:, i:] - shift_right)
        ssd = ndimage.filters.convolve(raw_ssd, kernel, mode='constant', cval=0)
        label_min = ssd < min_ssd[:, i:] 
        min_ssd[:, i:][label_min] = ssd[label_min]
        labels[:, i:][label_min] = i

    return labels

def Loopy_BP_(left, right, max_disp=30, max_iter=50, tau=10, smooth_weight=10):
    start = time.time()
    max_disp = max_disp
    max_iter = max_iter
    tau = tau
    left = left.astype(float)
    right = right.astype(float)
    smooth_weight = smooth_weight
    h, w = left.shape[:2]
    
    ####### Initialization
    
    ##empty messages
    m_UP = np.zeros((h, w, max_disp))
    m_DOWN = np.zeros((h, w, max_disp))
    m_LEFT = np.zeros((h, w, max_disp))
    m_RIGHT = np.zeros((h, w, max_disp))  
    ## Energy
    energy = np.zeros((max_iter))
    
    ## DataCost
    data_cost = np.zeros((h, w, max_disp))
    ## Beliefs
    beliefs = data_cost.copy()
           
    #### For Message Constant            
    m_UP_constant = np.zeros((h, w, max_disp))
    m_DOWN_constant = np.zeros((h, w, max_disp))
    m_LEFT_constant = np.zeros((h, w, max_disp))
    m_RIGHT_constant = np.zeros((h, w, max_disp))
    in_m_UP_constant = np.roll(m_DOWN_constant,  1, axis=0)
    in_m_DOWN_constant = np.roll(m_UP_constant, -1, axis=0)
    in_m_LEFT_constant = np.roll(m_RIGHT_constant,  1, axis=1)
    in_m_RIGHT_constant = np.roll(m_LEFT_constant, -1, axis=1)
    
    
    ###### Calculate Data Cost  
    for d in range(max_disp):
        data_cost[:, : , d] = np.minimum( norm(left - np.roll(right, d, axis=1), axis=2, ord=1), tau * np.ones((h, w)))

    
    ###### Update Message 
    def update_messages(m_UP, m_DOWN, m_LEFT, m_RIGHT):
        
        in_m_UP_constant = np.roll(m_DOWN,  1, axis=0)
        in_m_DOWN_constant = np.roll(m_UP, -1, axis=0)
        in_m_LEFT_constant = np.roll(m_RIGHT,  1, axis=1)
        in_m_RIGHT_constant = np.roll(m_LEFT, -1, axis=1)
        # npX -- neighbours excluding X
        np_UP = data_cost + in_m_LEFT_constant + in_m_DOWN_constant + in_m_RIGHT_constant
        np_DOWN = data_cost + in_m_UP_constant + in_m_LEFT_constant + in_m_RIGHT_constant
        np_LEFT = data_cost + in_m_UP_constant + in_m_DOWN_constant + in_m_RIGHT_constant
        np_RIGHT = data_cost + in_m_UP_constant + in_m_DOWN_constant + in_m_LEFT_constant

        sp_UP = np.amin(np_UP, axis=2)
        sp_DOWN = np.amin(np_DOWN, axis=2)
        sp_LEFT = np.amin(np_LEFT, axis=2)
        sp_RIGHT = np.amin(np_RIGHT, axis=2)

        for d in range(max_disp):
            m_UP[:, :, d] = np.minimum(np_UP[:, :, d], sp_UP + smooth_weight)
            m_DOWN[:, :, d] = np.minimum(np_DOWN[:, :, d], sp_DOWN + smooth_weight)
            m_LEFT[:, :, d] = np.minimum(np_LEFT[:, :, d], sp_LEFT + smooth_weight)
            m_RIGHT[:, :, d] = np.minimum(np_RIGHT[:, :, d], sp_RIGHT + smooth_weight)
            
        # normalization
        m_UP -= np.mean(m_UP, axis=2)[:, :, np.newaxis]
        m_DOWN -= np.mean(m_DOWN, axis=2)[:, :, np.newaxis]
        m_LEFT -= np.mean(m_LEFT, axis=2)[:, :, np.newaxis]
        m_RIGHT -= np.mean(m_RIGHT, axis=2)[:, :, np.newaxis]
        
        return m_UP, m_DOWN, m_LEFT, m_RIGHT
    
    ###### Compute Belief    
    def compute_beliefs(beliefs):
        in_m_UP = np.roll(m_DOWN,  1, axis=0)
        in_m_DOWN = np.roll(m_UP, -1, axis=0)
        in_m_LEFT = np.roll(m_RIGHT,  1, axis=1)
        in_m_RIGHT = np.roll(m_LEFT, -1, axis=1)
        beliefs += in_m_UP + in_m_DOWN + in_m_LEFT + in_m_RIGHT        
        return beliefs
        
    ###### Calculate Energy    
    def calc_energy(labels):
        E = 0
        hh, ww = np.meshgrid(range(h), range(w), indexing='ij')
        Ddp = data_cost[hh, ww, labels]
        E = np.sum(Ddp)
        int_cost__UP = smooth_weight * (labels - np.roll(labels,  1, axis=0) != 0)
        int_cost__LEFT = smooth_weight * (labels - np.roll(labels,  1, axis=1) != 0)
        int_cost__DOWN = smooth_weight * (labels - np.roll(labels, -1, axis=0) != 0)
        int_cost__RIGHT = smooth_weight * (labels - np.roll(labels, -1, axis=1) != 0)
        # set boundary costs to zero
        int_cost__UP[0, :] = 0
        int_cost__LEFT[:, 0] = 0
        int_cost__DOWN[-1, :] = 0
        int_cost__RIGHT[:, -1] = 0

        E += np.sum(int_cost__UP) + np.sum(int_cost__LEFT) + np.sum(int_cost__DOWN) + np.sum(int_cost__RIGHT)
        return E
    
    ######## SOLVE
    ## 0. Initializations - Done Above
    ## 1. Calculate Data Cost - Done above
    ## 2. Loop through iterations
    
    for i in range(max_iter):
        # print(f'Iteration {i}')
        m_UP, m_DOWN, m_LEFT, m_RIGHT = update_messages(m_UP, m_DOWN, m_LEFT, m_RIGHT)
        beliefs = compute_beliefs(beliefs)
        labels = np.argmin(beliefs, axis=2)
        energy[i] = calc_energy(labels)
        # if i == 10 or i%30 == 0:
        #     lbp_time = time.time() - start
        #     print(lbp_time)        
        #     labels = disp2jet(labels)
        #     labels = cv2.cvtColor(labels, cv2.COLOR_BGR2RGB)
        #     cv2.imwrite('./myoutput/' + 'Backpack-imperfect' + '_iter_loopyBP_' + str(i) + '.png', labels )

    return labels, energy

        
