#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 5/8/2020
# Trying the general code with complete dataset file for all videos


# In[ ]:


''' 
* Mobile_cif:
  300 frames 
  352x288
  352/32 = 11 , 288/32 = 9
  9*11 = 99 blocks/frame
  total no. of blocks = 99*300 = 29700

* Bus_cif:
  150 frames
  352x288
  11 , 9
  99*150 = 14850
  
  Then cif yuv will have 99 blocks per frame and a total of 99*no.of frames blocks per yuv
  
* 1080_YUV 1920x1080
  1920/32 = 60
  1080/32 = 33.75 = 33
  60*33
  Total no.of blocks/frame = 60*33.75 = 1980
  Then HD yuv will have 1980 blocks per frame and a total of 1980*no.of frames blocks per yuv

* 4K_YUV 3480x2160
  3480/32 = 108.75 = 108
  2160/32 = 67.5 = 67
  108*67
  Total no.of blocks/frame = 108*67 = 7236
  Then 4K yuv will have 7236 blocks per frame and a total of 7236*no.of frames blocks per yuv
  
'''
''' 
* Racing_HD yuv 1920x1080:
  600 frames --> Select only 60 frames
  1920/32 = 60 , 1080/32 = 33.75 = 33
  60*30 = 1980 blocks/frame
  total no. of blocks or modes = 1980*60 = 118800
'''


# In[7]:


# Important Libraries
import os
import cv2
import csv
import stat
import glob
import shutil
import operator
import numpy as np
import pandas as pd 
from PIL import Image
from random import choice
from numpy import savetxt
from numpy import loadtxt
from functools import reduce
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.collections import LineCollection
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Functions that calculates the size of certain folder
size = 0
path_ = ""
def calculate(path=os.environ["SYSTEMROOT"]):
    global size, path_
    size = 0
    path_ = path

    for x, y, z in os.walk(path):
        for i in z:
            size += os.path.getsize(x + os.sep + i)

def cevir(x):
    global path_
    size_in_megabytes =  x/1048576
    return size_in_megabytes


# In[5]:


'''
yuv_size: total no.frames of yuv 
frame_h: frame height (y)
frame_w: frame width (x)
blk_h: block height (y)
blk_w: block width (x)
total_blks: total blocks per yuv
blks_per_frame: total blocks per frame
'''
# Read CSV 
def read(file_path):
    return (pd.read_csv(file_path, header = None))

def calc_total_blocks(yuv_size, frame_h, frame_w):
    blk_h = int(frame_h/32)
    blk_w = int(frame_w/32)
    blks_per_frame = blk_h * blk_w
    total_blks = blks_per_frame * yuv_size
    
    return total_blks

# Creating Modes folders (35 folder)
def create_modes(path):
    for i in range(0,35):
        mode_path = os.path.join(path, str(i)) 
        os.mkdir(mode_path) 
        print("Directory '% s' created" % mode_path) 


# Form (x1,x2) (y1,y2) pairs to get horizontal and vertical block ranges
def merge(list1, list2): 
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))] 
    return merged_list 


# In[6]:


'''
yuv_name: i.e mob, bus, racing, race
frames_folders: folder containing extracted frames from yuv
blks_per_frame: no. of blocks per each frame according to yuv size
'''

# Drawing and Saving Figures for frames with modes
def drawing_modes(frames_folder, yuv_name, blks_per_frame):
    for k in range(0,int(len(mode)/blks_per_frame)):
        image = Image.open("E:/GP/Python_Codes-Data_preprocessing/General/" + str(frames_folder) +  "/" + str(k+1) + ".jpg")
        
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 5)
        ax.imshow(image)
            
        for i in range(blks_per_frame*k,blks_per_frame*(k+1)):
            for j in range(0,1):
                ax.axhline(y = new_y[i][j], xmin = new_x[i][0], xmax = new_x[i][1], color='y')
                ax.axvline(x = new_x[i][j], ymin = new_y[i][0], ymax  = new_y[i][1], color='y')
                
                graph = np.array(ax.annotate(str(mode[i]),
                                 xy = (new_x[0][0] , new_x[0][1]), 
                                 xytext =(int(new_x[i][1] - 25), int(new_y[i][1]) - 12), 
                                 color='y', 
                                 fontweight='bold', 
                                 fontsize=12))
                  
        fig.savefig('E:/GP/Python_Codes-Data_preprocessing/General/' + str(yuv_name) + '_Partitioned_Figures/' +                     str(yuv_name) + '_32x32_partitoned_F' + str(k + 1) + '.png')


# In[4]:


# Creating 9x11 mode matrix for each frame 
'''
* To make the matrix for a single frame we need to 0-blks_per_frame loops so to make for the 300 frames we need an additional 
  0-yuv_size loop
'''
def modes_matrix(yuv_size, blks_per_frame, yuv_h, y1): 
    sli = defaultdict(list)
    frame = defaultdict(list)
    
    # Creating an empty list consisting of 300 empty lists (list of lists)
    for i in range(1,4):
        nframe["new_frame{}".format(str(i))]  
        list(nframe.values())[i-1] = [[]*l for l in range(0,yuv_size)]
    
    # Creating an empty list consisting of yuv_size empty lists (list of lists)
    '''
    instead we can create list of lists according to yuv_h:
    list contain all slice lists (without naming) then each list consists of lists their size depends on yuv_h
    '''
    slice = [[]*i for i in range(0,yuv_h)] # No of slice lists depends on yuv_h
    for j in range(0,yuv_h):
        slice[j] = [[]*i for i in range(0,yuv_h)]
    
    for i in range(1,yuv_h + 1):
        sli["slice{}".format(str(i))] 
        list(sli.values())[i-1].append([[]*l for l in range(0,yuv_size)])
    
    '''
    if conditions: 9 or 33 or 60 --> according to yuv_h
    y1[j] - step32 stop at 2080 --> depends on blks_per_frame
    '''
    for k in range(0, yuv_size):  # Each 352x288 frame
        for j in range(blks_per_frame*k,blks_per_frame*(k+1)): # Each 32x32 frame where each has 99 blocks
            for i in range(0,blks_per_frame,32):
                if(y1[j] == i):
                    list(sli.values())[i][0][k].append(mode[j])
        
        for r in range(0,yuv_h):
            list(nframe.values())[0] += list(sli.values())[r]
            
        new_frame[k] = slice1[k] + slice2[k] + slice3[k] + slice4[k] + slice5[k] + slice6[k] + slice7[k] + slice8[k] + slice9[k]
        new_frame[k] = np.array(frame[k]).reshape(9,11)
            
        savetxt('Mode_Matrices/Mode_matrix' + str(k+1) + '.csv', new_frame[k], fmt='%d', delimiter=',')
        savetxt('Mode_Lists/Mode_list' + str(k+1) + '.csv', frame[k], fmt='%d', delimiter=',')


# In[8]:


# Another solution to general mode matrix function
# Creating 9x11 mode matrix for each frame 
'''
* To make the matrix for a single frame we need to 0-blks_per_frame loops so to make for the 300 frames we need an additional 
  0-yuv_size loop
'''
def modes_matrix(yuv_name, yuv_size, blks_per_frame, yuv_h, yuv_w, y1): 
    frame = [[]*yuv_size for i in range(0,yuv_size)]
    new_frame = [[]*300 for i in range(0,yuv_size)]
    
    '''
    instead we can create list of lists according to yuv_h:
    list contain all slice lists (without naming) then each list consists of lists their size depends on yuv_h
    '''
    row = [[]*i for i in range(0,yuv_h)] # No of slice lists depends on yuv_h
    for j in range(0,yuv_h):
        row[j] = [[]*i for i in range(0,yuv_size)]
    
    '''
    if conditions: 9 or 33 or 60 --> according to yuv_h
    y1[j] - step32 stop at 2080 --> depends on blks_per_frame
    '''
    for k in range(0, yuv_size):  # Each 352x288 frame
        for j in range(blks_per_frame*k,blks_per_frame*(k+1)): # Each 32x32 frame where each has 99 blocks
            for i in range(0,blks_per_frame,32):
                for t in range(0, yuv_h):
                    if(y1[j] == i):
                        row[t][k].append(mode[j])
        
        for g in range(0,yuv_h):   
            frame[k] += row[g][k]
        
        new_frame[k] = np.array(frame[k]).reshape(yuv_h,yuv_w) 
            
        savetxt('E:/GP/Python_Codes-Data_preprocessing/General/' + str(yuv_name) + '_Mode_Matrices/' + str(yuv_name) +                 '_Mode_matrix' + str(k+1) + '.csv', new_frame[k], fmt='%d', delimiter=',')
        
                    
        savetxt('E:/GP/Python_Codes-Data_preprocessing/General/' + str(yuv_name) + '_Mode_Lists/' + str(yuv_name) +                 '_Mode_list' + str(k+1) + '.csv', frame[k], fmt='%d', delimiter=',')


# In[14]:


def mode_splitting():
    source = []
    destination = []
    dest_paths = []
    l = []
    
    # Loop for getting path for each of the 300 frames-blocks
    for j in range(0,300):
        Frame_path = "E:/GP/Python_Codes-Data_preprocessing/Mode_Distribution/Mobile_Frames_32x32/frame" + str(j+1) + "-"
        mList = pd.read_csv("Mode_Lists/Mode_list" + str(j+1) + ".csv", header = None)
        l.append(mList)
        
        # Rejoin modes of 300 frames into 1 csv file after reshaping
        vid_modes = pd.concat(l, axis=0, ignore_index=True)
        vid_modes = vid_modes[0].to_list()
        # savetxt("full_modes.csv", vid_modes, fmt='%d', delimiter=',')

        # Completing the path for each of the 300 frames-blocks
        for h in range(0,99):
            source.append(Frame_path + str(h) + ".jpg")
    
    for i in range(0,29700):
        destination.append("Data/" + str(vid_modes[i]))
        dest_paths.append("Block" + str(i+1) + "-" + str(i) + " copied to folder " + str(destination[i]))
        dest = shutil.copy(source[i], destination[i])
        
    # Calculate size of Data folder and if blocks is copied in it, save operation
    calculate("E:/GP/Python_Codes-Data_preprocessing/Mode_Distribution/Data")
    s = cevir(size)
    if(s != 0):
        savetxt("Block_copy.txt", dest_paths, fmt='%s', delimiter=',')


    
    # print(len(destination))
    # print(len(source))
    # print(dest_paths) 
    savetxt("source_paths.txt", source, fmt='%s', delimiter=',')
    savetxt("dst_paths.txt", destination, fmt='%s', delimiter=',')


# In[5]:


# Reading Data
mob = "Srch_dpth-X-Y-Mode_mob_32x32.csv"
bus = "Srch_dpth-X-Y-Mode_Bus_32x32.csv"

mob_read = read(mob)
bus_read = read(bus)

# Takeimportant features from file (pelx1, pelx2, pely1, pely2, mode)
x1 = mob.iloc[:, 1]
x2 = mob.iloc[:, 2]
y1 = mob.iloc[:, 3]
y2 = mob.iloc[:, 4]
mode = mob.iloc[:, 5]

x = merge(x1,x2)
y = merge(y1,y2)

# Convert pairs from list to numpy array
new_y = np.array([list(map(int,s)) for s in y])
new_x = np.array([list(map(int,s)) for s in x])

for i in range(1,301):
    image = Image.open("E:/GP/Python_Codes-Data_preprocessing/Mode_Distribution/Mobile_CIF_Frames/" + str(i) + ".jpg")
#     drawing_modes(image)


# In[46]:


# Divide mode into lists, list for each frame after reshaping 
modes_matrix()


# In[21]:


# [Path] to 32x32 blocks that will be divided according to mode
# Give the path of each frame in yuv and call the function to move images in each folder to its corresponding
# mode folders 

# Creating an empty list consisting of 300 list (list of lists)
mode_list = [[]*300 for i in range(0,300)]

# Dividing the full 300 frames modes dataset csv file into 300 csv files one for each frame
for i in range(0,300):
    mode_list[i] = pd.read_csv("Mode_Lists/Mode" + str(i) + ".csv", header = None)
    mode_list[i] = mode_list[i].values.tolist()
    savetxt('Mode_CSV_Files/Frame' + str(i+1) + '_modes.csv', mode_list[i], fmt='%d', delimiter=',')

# mode_list[1]


# In[ ]:


# Creating mode folders
path = "E:/GP/Python_Codes-Data_preprocessing/Mode_Distribution/Data"
create_modes(path)

# Moving data from images folder to mode folders
mode_splitting()


# In[13]:


mode_splitting()


# In[160]:


mode.shape


# In[184]:


# Check the final size of Data file
calculate("E:/GP/Python_Codes-Data_preprocessing/Mode_Distribution/Data")
s = cevir(size)
s


# In[54]:


creating_train_test_val()


# In[143]:




