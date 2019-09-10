savePath = r'\\stephcurry\\Users\\nlab\\Desktop\\prey_capture_analysis' #path to save out data

import os
import cv2
import numpy as np
import h5py
from tkinter import Tk
from tkinter.filedialog import askopenfilename

animal = input('animal ID (all uppercase): ')
expdate = input('experiment date (MMDDYY): ')
condition = input('condition (control or enriched): ')

ftypes = [('Choose movie file', '*.avi')] #select file type you want to load
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename(filetypes = ftypes) # show an "Open" dialog box and return the path to the selected file
base_dir, vidname = os.path.split(filename)
os.chdir(savePath)
print(filename)


vid = cv2.VideoCapture(filename)
def nothing(x):
	pass

total_frames_1 = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) #get the total number of frames
fps = int(vid.get(cv2.CAP_PROP_FPS)) #get the video framerate

# cv2.namedWindow('image')
cv2.namedWindow('image',cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('image',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

# create trackbars for frame change
cv2.createTrackbar('Frame','image',0,total_frames_1-1,nothing)
font = cv2.FONT_HERSHEY_DUPLEX
fontcol = (0,0,0)
fontsize = 0.6

short_appr_start = []
long_appr_start = []
explore_start = []
short_appr_end = []
long_appr_end = []
explore_end = []
last_frame = 0

#the loop where all logging happens
while(1):
	# get current positions of four trackbars
	n_frame = cv2.getTrackbarPos('Frame','image')

	vid.set(cv2.CAP_PROP_POS_FRAMES, n_frame)

	ret, frame_1 = vid.read()

	k = cv2.waitKey(1) 
	if k == 27: #esc key
	    break

	elif k == ord('i'): # Left key fast
	    n_frame = cv2.setTrackbarPos('Frame','image',n_frame-5)
	elif k == ord('o'): # Right key fast
	    n_frame = cv2.setTrackbarPos('Frame','image',n_frame+5)
	elif k == ord('k'): # Left key
	    n_frame = cv2.setTrackbarPos('Frame','image',n_frame-1)
	elif k == ord('l'): # Right key
	    n_frame = cv2.setTrackbarPos('Frame','image',n_frame+1)

	elif k==ord('q'):
	    short_appr_start.append(n_frame)
	elif k==ord('w'):
	    short_appr_end.append(n_frame)
	elif k==ord('a'):
	    long_appr_start.append(n_frame)
	elif k==ord('s'):
	    long_appr_end.append(n_frame)
	elif k==ord('z'):
	    explore_start.append(n_frame)
	elif k==ord('x'):
	    explore_end.append(n_frame)

	elif k==ord('r'):
	    short_appr_start.pop(-1)
	elif k==ord('t'):
	    short_appr_end.pop(-1)
	elif k==ord('f'):
	    long_appr_start.pop(-1)
	elif k==ord('g'):
	    long_appr_end.pop(-1)
	elif k==ord('v'):
	    explore_start.pop(-1)
	elif k==ord('b'):
	    explore_end.pop(-1)

	elif k == ord('1'): #for if the gui crashes
		last_frame = n_frame
		cv2.destroyAllWindows()
		cv2.namedWindow('image',cv2.WND_PROP_FULLSCREEN)
		cv2.setWindowProperty('image',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
		cv2.createTrackbar('Frame','image',0,total_frames_1-1,nothing) # create trackbars for frame change
		vid.set(cv2.CAP_PROP_POS_FRAMES, last_frame)

	height, width, channels = frame_1.shape
	cv2.putText(frame_1, str('short %d,%d long %d,%d explore %d,%d' % (len(short_appr_start),len(short_appr_end),
		len(long_appr_start),len(long_appr_end),len(explore_start),len(explore_end))), (int(width*0.76),int(height*0.05)), font, fontsize, fontcol, 1, cv2.LINE_AA)
	cv2.putText(frame_1, 'add start/end short(q,w) long(a,s) explore(z,x)', (int(width*0.76),int(height*0.1)), font, fontsize, fontcol, 1, cv2.LINE_AA)
	cv2.putText(frame_1, 'remove start/end short(r,t) long(f,g) explore(v,b)', (int(width*0.76),int(height*0.15)), font, fontsize, fontcol, 1, cv2.LINE_AA)
	cv2.putText(frame_1, 'i <--- ---> o | 1:if crashes', (int(width*0.76),int(height*0.2)), font, fontsize, fontcol, 1, cv2.LINE_AA)
	cv2.putText(frame_1, str('k <- -> l | crashed frame %d' % (last_frame)), (int(width*0.76),int(height*0.25)), font, fontsize, fontcol, 1, cv2.LINE_AA)
	cv2.imshow('image',frame_1)

#close the gui
cv2.destroyAllWindows()

#convert the data into numpy arrays
short_appr_start = np.array(short_appr_start)
short_appr_start = np.array(short_appr_end)
long_appr_start = np.array(long_appr_start)
long_appr_end = np.array(long_appr_end)
explore_start = np.array(explore_start)
explore_end = np.array(explore_end)


with h5py.File(vidname[:-4] + ".hdf5", "a") as f:
	try:
		grp = f.create_group('behav_events')
		grp.create_dataset("short_appr_start",data=short_appr_start, dtype='i')
		grp.create_dataset("short_appr_end",data=short_appr_end, dtype='i')
		grp.create_dataset("long_appr_start",data=long_appr_start, dtype='i')
		grp.create_dataset("long_appr_end",data=long_appr_end, dtype='i')
		grp.create_dataset("explore_start",data=explore_start, dtype='i')
		grp.create_dataset("explore_end",data=explore_end, dtype='i')

		grp = f.create_group('experiment_info')
		grp.attrs['expdate']=np.string_(expdate)
		grp.attrs['animal']=np.string_(animal)
		grp.attrs['condition']=np.string_(condition)
		grp.create_dataset('fps', data=fps, dtype = 'i')
	except:
		del f["behav_events"]
		del f["experiment_info"]
		grp = f.create_group('behav_events')
		grp.create_dataset("short_appr_start",data=short_appr_start, dtype='i')
		grp.create_dataset("short_appr_end",data=short_appr_end, dtype='i')
		grp.create_dataset("long_appr_start",data=long_appr_start, dtype='i')
		grp.create_dataset("long_appr_end",data=long_appr_end, dtype='i')
		grp.create_dataset("explore_start",data=explore_start, dtype='i')
		grp.create_dataset("explore_end",data=explore_end, dtype='i')

		grp = f.create_group('experiment_info')
		grp.attrs['expdate']=np.string_(expdate)
		grp.attrs['animal']=np.string_(animal)
		grp.attrs['condition']=np.string_(condition)
		grp.create_dataset('fps', data=fps, dtype = 'i')