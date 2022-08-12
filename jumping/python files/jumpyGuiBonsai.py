### jumpyGuiBonsai.py
### manually label trial events from bonsai videos

#set the scrub rates for moving throug the video
fastscrub = 0.4
slowscrub = 0.02

##### import dependencies
import os, fnmatch
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm #from tqdm import tqdm_notebook as tqdm
import matplotlib as mpl
mpl.rcParams.update({'font.size': 22})
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory, Directory
import json

##### define functions
#a function to find the files we want
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files: 
            if fnmatch.fnmatch(name,pattern): 
                result.append(os.path.join(root,name))
    if len(result)==1:
        result = result[0]
    return result

#function to get correct frame in case there's lag in the acquisition
def find_first(item, vec):
    return np.argmin(np.abs(vec-item))

#function to load Bonsai timestamps
def load_Bonsai_TS(file_name):
    TS_read = pd.read_csv(file_name, header=None)
    ts = list(TS_read[0])
    return ts

def hms_to_seconds(t):
    h = int(t.split(':')[0])
    m = int(t.split(':')[1])
    s = float(t.split(':')[2])
    return 3600*h + 60*m + s

#function to save time stamp values for trial events
def save_vidclip_ts(df,file_name):
    df.to_json(file_name)

#####

#pick experiment path, get relevant files
root = Tk()
root.withdraw() # we don't want a full GUI, so keep the root window from appearing
expt_path = askdirectory(title='Choose experiment folder') # show an "Open" dialog box and return the path to the selected file
print('you have selected: ', expt_path)

#load trial data
trial_data_file = find('*TrialData.csv',expt_path)
try:
    columns = ['trial','success','platform','distance','laser']
    trial_data = pd.read_csv(trial_data_file,header=None,names=columns)
except:
    columns = ['trial','success','platform','distance']
    trial_data = pd.read_csv(trial_data_file,header=None,names=columns)
ntrials = trial_data.shape[0]

#load video, timestamps
top_vid_files = find('*TOP*.avi',expt_path)
side_vid_files = find('*SIDE*.avi',expt_path)
top_ts_files = find('*TOP_BonsaiTS*.csv',expt_path)
side_ts_files = find('*SIDE_BonsaiTS*.csv',expt_path)
top_vid_files.sort(key=os.path.getmtime)
side_vid_files.sort(key=os.path.getmtime)
top_ts_files.sort(key=os.path.getmtime)
side_ts_files.sort(key=os.path.getmtime)

# create vidclip timestamp file
vidclip_ts = trial_data_file[:-13] + '_vidclip_ts.txt'

# expt_name = trial_data.loc[0,'expdate'] + '_' + trial_data.loc[0,'subject']
### create datarame for frame times for trial events, or load if it already exists
if os.path.exists(vidclip_ts):
    ts_data = pd.read_json(vidclip_ts)
    # trial_num = sum(ts_data['Top_End']>0)-1 #temporarily commenting out to go back through experiments 05/04/22
else:
    columns = ['Top_Start','Top_Jump','Top_End','Side_Start','Side_Jump','Side_End']
    ts_data = pd.DataFrame(columns=columns, index=[n for n in np.arange(ntrials)])
trial_num = 0 #temporarily out of if loop to start at trial 0 05/04/22

print('there are %d trials' % ntrials)

breaker = False
while(True):

    if breaker:
        break
    #load the video time stamps
    top_ts = load_Bonsai_TS(top_ts_files[trial_num])
    side_ts = load_Bonsai_TS(side_ts_files[trial_num])
    top = np.array([hms_to_seconds(t) for t in top_ts])
    side = np.array([hms_to_seconds(t) for t in side_ts])
    
    #load the side video
    side_vid = side_vid_files[trial_num]
    vid = cv2.VideoCapture(side_vid)
    tot_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vid.get(cv2.CAP_PROP_FPS)

    #create the GUI window
    cv2.namedWindow('image')#,cv2.WND_PROP_FULLSCREEN)

    # create trackbars for frame change
    def nothing(x):
        pass
    cv2.createTrackbar('Frame','image',0,tot_frames-1,nothing)
    font = cv2.FONT_HERSHEY_DUPLEX
    fontcol = (0,255,0)
    fontsize = 0.5
    lastdisp = ''

    #loop for using key presses to log trial events
    while(True):

        # get current positions of trackbar
        n_frame = cv2.getTrackbarPos('Frame','image')
        vid.set(cv2.CAP_PROP_POS_FRAMES, n_frame)

        ret, frame_1 = vid.read()
        
        k = cv2.waitKey(1) 
        
        #esc key exits
        if k == 27:
            breaker = True
            break
        #c key reloads GUI in case it freezes
        elif k == ord('c'):
            cv2.destroyAllWindows()
            cv2.namedWindow('image',cv2.WND_PROP_FULLSCREEN)
            cv2.createTrackbar('Frame','image',0,tot_frames-1,nothing) # create trackbars for frame change
            vid.set(cv2.CAP_PROP_POS_FRAMES, n_frame)

        #add events
        elif k==ord('q'):
            ts_data.loc[trial_num,'Side_Start'] = int(n_frame)
            ts_data.loc[trial_num,'Top_Start'] = int(find_first(side[n_frame],top))
            lastdisp = '%d Start (side=%d,top=%d)' % (trial_num+1,ts_data.loc[trial_num,'Side_Start'],
                ts_data.loc[trial_num,'Top_Start'])
            save_vidclip_ts(ts_data,vidclip_ts)
        elif k == ord('w'):
            ts_data.loc[trial_num,'Side_Jump'] = int(n_frame)
            ts_data.loc[trial_num,'Top_Jump'] = int(find_first(side[n_frame],top))
            lastdisp = '%d Jump (side=%d,top=%d)' % (trial_num+1,ts_data.loc[trial_num,'Side_Jump'],
                ts_data.loc[trial_num,'Top_Jump'])
            save_vidclip_ts(ts_data,vidclip_ts)
        elif k == ord('e'):
            ts_data.loc[trial_num,'Side_End'] = int(n_frame)
            ts_data.loc[trial_num,'Top_End'] = int(find_first(side[n_frame],top))
            lastdisp = '%d End (side=%d,top=%d)' % (trial_num+1,ts_data.loc[trial_num,'Side_End'],
                ts_data.loc[trial_num,'Top_End'])
            save_vidclip_ts(ts_data,vidclip_ts)

        #navigation keys
        elif k == ord('a'): # current trial start
            n_frame = cv2.setTrackbarPos('Frame','image',int(ts_data.loc[trial_num,'Side_Start']))
        elif k == ord('s'): # current jump start
            n_frame = cv2.setTrackbarPos('Frame','image',int(ts_data.loc[trial_num,'Side_Jump']))
        elif k == ord('d'): # current jump end
            n_frame = cv2.setTrackbarPos('Frame','image',int(ts_data.loc[trial_num,'Side_End']))

        elif k == ord('i'): # Left key fast
            n_frame = cv2.setTrackbarPos('Frame','image',n_frame-int(round(fps*fastscrub)))
        elif k == ord('o'): # Right key fast
            n_frame = cv2.setTrackbarPos('Frame','image',n_frame+int(round(fps*fastscrub)))
        elif k == ord('k'): # Left key
            n_frame = cv2.setTrackbarPos('Frame','image',n_frame-int(round(fps*slowscrub)))
        elif k == ord('l'): # Right key
            n_frame = cv2.setTrackbarPos('Frame','image',n_frame+int(round(fps*slowscrub)))

        elif k == ord('n'): #next jump
            if trial_num<ntrials-1:
                trial_num += 1
                break
        elif k == ord('b'): #back one jump
            if trial_num>0:
                trial_num -= 1
                break

        cv2.putText(frame_1, 'jump %d/%d Start %d Jump %d End %d' % (trial_num+1,ntrials,
            sum(ts_data['Top_Start']>0),sum(ts_data['Top_Jump']>0),sum(ts_data['Top_End']>0)),
            (10,20), font, fontsize, fontcol, 1, cv2.LINE_AA)
        cv2.putText(frame_1, 'n:next b:back  q-w-e: +times', (10,40), font, fontsize, fontcol, 1, cv2.LINE_AA)
        cv2.putText(frame_1, 'a-s-d: go to curren times', (10,60), font, fontsize, fontcol, 1, cv2.LINE_AA)
        cv2.putText(frame_1, 'i <------> o | k <--> l  c:if crashes', (10,80), font, fontsize, fontcol, 1, cv2.LINE_AA)
        if trial_num>=0:
            cv2.putText(frame_1, 'plat %d dist %d outcome %d' % (trial_data.loc[trial_num,'platform'],
                trial_data.loc[trial_num,'distance'],trial_data.loc[trial_num,'success']),
                (10,100), font, fontsize, fontcol, 1, cv2.LINE_AA)
        cv2.putText(frame_1, '%s' % (lastdisp), (10,120), font, fontsize, fontcol, 1, cv2.LINE_AA)
        cv2.imshow('image',frame_1)

    #close the gui
    cv2.destroyAllWindows()
    try:
        tsdata = tsdata.astype(int)
    except:
        pass
    save_vidclip_ts(ts_data,vidclip_ts)
print('completed %d/%d trials' % (sum(ts_data['Top_Start']>0),ntrials))