# analyze_movements.py

### This is the backend for all of the analysis and plotting of movement data
### using the dataframe created by get_all_movements

### IMPORT DEPENDENCIES

###basics
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib as mpl
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.pyplot import cm

# %matplotlib inline
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
# %matplotlib notebook
# import umap
from scipy import stats,signal,misc,integrate
import pylab
import seaborn as sns
import math
import cv2
import os,fnmatch
from itertools import product, compress
import itertools
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
import random
import json
import statsmodels.api as sm
from statsmodels.formula.api import ols

###bokeh plotting
# from bokeh.io import output_file, show
# from bokeh.layouts import gridplot
# from bokeh.models import ColumnDataSource
# from bokeh.models import LinearColorMapper
# from bokeh.plotting import figure, show, output_notebook
# from bokeh.models import Circle,MultiLine
# from bokeh.models import Title
# from bokeh.layouts import grid, column, row
# from bokeh.transform import linear_cmap
# from bokeh.palettes import Spectral,Turbo256
import itertools

# output_notebook()

###pca and k-means

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import mixture
from sklearn.linear_model import LinearRegression


### FUNCTIONS

def aborts_as_failures(og_df):
    df = og_df.copy()
    df = df.astype({'success': 'int32'})#,'jumpdist':np.float64})
    df[df['success']==2] = df[df['success']==2].replace(2,0)
    
    return df

def remove_aborts(og_df):
    df = og_df.copy()
    df = df.astype({'success': 'int32'})#,'jumpdist':np.float64})
    df = df[df['success']!=2]
    df.reset_index(inplace=True,drop=True)
    
    return df

def num_seq_in_list(numbers,dur,fps):
    #numbers: list of state labels
    #dur: minimum state duration you want
    #fps: the framerate
    goods = []
    count = 0
    for i in range(len(numbers)-1):
        if numbers[i] + 1 == numbers[i+1]:
            if goods == []:
                goods.append([numbers[i]])
                count = count + 1
            elif numbers[i] != goods[-1][-1]:
                goods.append([numbers[i]])
                count = count + 1
            if numbers[i+1] != goods[-1]:
                goods[-1].extend([numbers[i+1]])
    goods = list(compress(goods,[len(g)>(int(dur*fps)) for g in goods]))
    n_seq = len(goods)
    return n_seq

def filter_decision_period(row,side,key,l_thresh,ts_down,time_per,st_pt,end_pt):
    box_sz = 5 #for box filtering
    box = np.ones(box_sz)/box_sz
    
    like = row[side + ' ' + key + ' likelihood']
    xtr = row[side + ' ' + key + ' x']
    ytr = row[side + ' ' + key + ' y']
    
    # xtr[like<l_thresh] = np.nan #nan out low likelihoods
    # xtr = pd.Series(xtr).interpolate().to_numpy() #interp over nans
    # xtr = pd.Series(xtr).fillna(method='bfill').to_numpy() #if first vals are nans fill them
    # xtr = signal.medfilt(xtr,kernel_size=3) #mean filter
    # xtr = np.convolve(xtr, box, mode='same') #smoothing filter
    
    # xtr = xtr[:row[side + '_Jump']-row[side + '_Start']] #commented out for revisions 061222
    xtr = xtr[st_pt:end_pt][-int(time_per*row['fps']):-1] #full trace
    xtr = xtr[::ts_down]
    
    # ytr = pd.Series(ytr).interpolate().to_numpy() #interp over nans
    # ytr = pd.Series(ytr).fillna(method='bfill').to_numpy() #if first vals are nans fill them
    # ytr = signal.medfilt(ytr,kernel_size=3) #mean filter
    # ytr = np.convolve(ytr, box, mode='same') #smoothing filter
    
    # ytr = ytr[:row[side + '_Jump']-row[side + '_Start']] #commented out for revisions 061222
    ytr = ytr[st_pt:end_pt][-int(time_per*row['fps']):-1] #full trace
    ytr = ytr[::ts_down]
    
    return xtr,ytr


def split_seqs(numbers):
    #numbers: list of state labels
    goods = []
    count = 0
    for i in range(len(numbers)-1):
#         print(i,goods,numbers[i])
        if numbers[i] + 1 == numbers[i+1]:
            if goods == []:
                goods.append([numbers[i]])
                count = count + 1
            elif numbers[i] != goods[-1][-1]:
                goods.append([numbers[i]])
                count = count + 1
            if numbers[i+1] != goods[-1][-1]:
                goods[-1].extend([numbers[i+1]])
    n_seq = len(goods)
    return n_seq, goods

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def plot_subset_all_states(df,K,side,st_list,n_rand,ts_down,time_per,st_pt,end_pt,vid_dir,ax_lim,save_pdf,pp):
    nrows = len(st_list)
    ncols = 1
    fac = 3
    fig, axs = plt.subplots(nrows,ncols,figsize=(fac*ncols,fac*nrows))
    axs = axs.ravel()

    plt_trials = np.random.randint(0,len(df), n_rand, dtype=int)
    plt_df = df.iloc[plt_trials]
    plt_df.reset_index(inplace=True,drop=True)
    for index,row in plt_df.iterrows():
        xtr,ytr = filter_decision_period(row,side,'LEye',0.95,ts_down,time_per,st_pt,end_pt)

        state_list = row['trMAPs'].copy()
        state_list[~row['trMasks']] = -1
        for s,st in enumerate(st_list):
            ax = axs[s]
            if index==0:
                fname = vidname_from_row(vid_dir,side,row)
                frame,fps,frame_width,frame_height, ret = grab_vidframe(fname,row[side + '_Jump']-row[side + '_Start']-2)
                ax.imshow(frame)
            st_inds = list(np.where(state_list==st)[0])
            n_seq, on_seq = split_seqs(st_inds)
            if n_seq>0:
                for seq in on_seq:
                    try:
                        l = len(seq)
                        jet = cm.get_cmap('jet',l)
                        new_cols = jet(np.linspace(0, 1, l))
                        # cmap = discrete_cmap(l,base_cmap=plt.cm.jet)
                        c = np.arange(l)
                        # p1 = ax.scatter(xtr[seq], ytr[seq], s=5, c=c, cmap=cmap, zorder=1)
                        for p in range(len(xtr[seq])-1):
                            ax.plot([xtr[seq[p]],xtr[seq[p+1]]],[ytr[seq[p]],ytr[seq[p+1]]],'-',color=new_cols[c[p],:],linewidth=2)
                    except:
                        continue
            if index==len(plt_df)-1:
                ax.axis(ax_lim)
                ax.axis('off')
                ax.set_title('state %s' % str(s+1))
                
    fig.tight_layout()
    
    if save_pdf:
        pp.savefig(fig)
        plt.close(fig)
        
    return fig, axs



def plot_hmm_freq(base_df,save_pdf,pp,suptitle=''):
    mn = base_df.groupby(['ocular','distance_DLC','success']).mean()
    mn.reset_index(inplace=True)
    sem = base_df.groupby(['ocular','distance_DLC','success']).std()/np.sqrt(len(np.unique(base_df['subject'])))
    sem.reset_index(inplace=True)

    fig, axs = plt.subplots(1,2,figsize=(10,5))
    axs = axs.ravel()
    for o,oc in enumerate(np.unique(base_df['ocular'])):
        ax = axs[o]
        mn_fail = mn[(mn['ocular']==oc)&(mn['success']==0)]
        mn_succ = mn[(mn['ocular']==oc)&(mn['success']==1)]
        sem_fail = sem[(sem['ocular']==oc)&(sem['success']==0)]
        sem_succ = sem[(sem['ocular']==oc)&(sem['success']==1)]
        ax.errorbar(x=mn_fail['distance_DLC'],y=mn_fail['hmm_freq'],
                    yerr=sem_fail['hmm_freq'],color=[0.5,0.5,0.5],label='failure')
        ax.errorbar(x=mn_succ['distance_DLC'],y=mn_succ['hmm_freq'],
                    yerr=sem_succ['hmm_freq'],color='k',label='success')
        ax.set_xlim(6,26)
        ax.set_ylim(0,15)
        ax.set_ylabel('frequency')
        ax.set_xlabel('gap distance (cm)')
        ax.set_title(oc)
        ax = xy_axis(ax)
    ax.legend()
    
    fig.suptitle(suptitle)
    if save_pdf:
        pp.savefig(fig)
        plt.close(fig)
        
def df_row_from_labels(df,labels_row):
    df_row = df[(df['subject']==labels_row['subject'])&\
                (df['expdate']==labels_row['expdate'])&\
                (df['trial']==labels_row['trial'])].iloc[0]
    return df_row

def labels_df_from_df_row(labels,df_row):
    labels_row = labels[(labels['subject']==df_row['subject'])&\
                (labels['expdate']==df_row['expdate'])&\
                (labels['trial']==df_row['trial'])]
    return labels_row



def plot_hmm_stats(df,condition,manipulation,save_pdf,pp,suptitle=''):
    gray = [0.5,0.5,0.5]
    leg_fs = 8
    plt_cols = [gray,'k']
    suc_lab = ['fail','success']
    hmm_lab = ['state absent','state present']
    fig, axs = plt.subplots(4,2,figsize=(10,20))
    axs = axs.ravel()
    for c,cond in enumerate(np.unique(df[condition])):
        cond_df = df[df[condition]==cond]
        cond_df.reset_index(inplace=True)
        for m,man in enumerate(np.unique(df[manipulation])):
            temp_df = cond_df[cond_df[manipulation]==man]
            temp_df.reset_index(inplace=True,drop=True)

            plot_df = remove_aborts(temp_df)
#             plot_df = remove_aborts(temp_df)

            mn = plot_df.groupby(['distance_DLC','success']).mean()
            sem = plot_df.groupby(['distance_DLC','success']).std()/np.sqrt(len(np.unique(plot_df['subject'])))
            mn.reset_index(inplace=True)
            sem.reset_index(inplace=True)
            ax = axs[c]
            for suc in np.unique(mn['success']):
                if m==1:
                    ax.errorbar(x=mn[mn['success']==suc]['distance_DLC'],
                                y=mn[mn['success']==suc]['hmm_freq'],
                                yerr=sem[sem['success']==suc]['hmm_freq'],color=plt_cols[suc],label=suc_lab[suc])
                else:
                    ax.errorbar(x=mn[mn['success']==suc]['distance_DLC'],
                                y=mn[mn['success']==suc]['hmm_freq'],
                                yerr=sem[sem['success']==suc]['hmm_freq'],color=plt_cols[suc])
            ax.set_ylim(0,15)
            ax.set_xlim(6,26)
            ax.set_ylabel('state frequency')
            ax.set_xlabel('gap distance (cm)')
            if m==1:
                ax.set_title(cond + ' state on %d%% trials' % int(100*temp_df.shape[0]/cond_df.shape[0]))
                ax.legend(fontsize=leg_fs)
                
            mn = plot_df.groupby(['distance_DLC']).mean()
            sem = plot_df.groupby(['distance_DLC']).std()/np.sqrt(len(np.unique(plot_df['subject'])))
            mn.reset_index(inplace=True)
            sem.reset_index(inplace=True)
            ax = axs[c+2]
            ax.errorbar(x=mn['distance_DLC'],y=mn['success'],yerr=sem['success'],color=plt_cols[m],label=hmm_lab[m])
            ax.set_ylim(0,1.1)
            ax.set_xlim(6,26)
            ax.set_ylabel('success')
            ax.set_xlabel('gap distance (cm)')
            if m==1:
                ax.legend(fontsize=leg_fs)

            plot_df = remove_aborts(temp_df)
            mn = plot_df.groupby(['distance_DLC']).mean()
            sem = plot_df.groupby(['distance_DLC']).std()/np.sqrt(len(np.unique(plot_df['subject'])))
            mn.reset_index(inplace=True)
            sem.reset_index(inplace=True)

            ax = axs[c+4]
            ax.errorbar(x=mn['distance_DLC'],y=mn['jumpdist'],yerr=sem['jumpdist'],color=plt_cols[m],label=hmm_lab[m])
            ax.set_ylim(6,26)
            ax.set_xlim(6,26)
            ax.set_ylabel('distance jumped (cm)')
            ax.set_xlabel('gap distance (cm)')
            ax.plot([6,26],[6,26],':',color=gray)
            if m==1:
                ax.legend(fontsize=leg_fs)
            
            ax = axs[c+6]
            ax.errorbar(x=mn['distance_DLC'],y=mn['accuracy'],yerr=sem['accuracy'],color=plt_cols[m],label=hmm_lab[m])
            ax.set_ylim(0,15)
            ax.set_xlim(6,26)
            ax.set_ylabel('error (cm)')
            ax.set_xlabel('gap distance (cm)')
            if m==1:
                ax.legend(fontsize=leg_fs)
    
    fig.suptitle(suptitle)
    for ax in axs:
        ax = xy_axis(ax)
    fig.tight_layout()
    if save_pdf:
        pp.savefig(fig)
        plt.close(fig)


def get_arhmm_start_frames(numbers):
    #numbers: list of state labels
    #dur: minimum state duration you want
    #fps: the framerate
    goods = []
    inds = []
    count = 0
    for i in range(len(numbers)-1):
        if numbers[i] == numbers[i + 1]:
            if goods == []:
                goods.append(numbers[i])
                inds.append(i)
                count = count + 1
            elif numbers[i] != goods[-1]:
                goods.append(numbers[i])
                inds.append(i)
                count = count + 1
            if numbers[i+1] != goods[-1]:
                goods[-1].extend([numbers[i+1]])
    n_seq = len(goods)
    goods = np.array(goods)
    inds = np.array(inds)
    return n_seq,goods,inds


def find(pattern, path):
    ### a function that returns the full file path in a list given a pattern and directory, includes subfolders
    result = []
    for root, dirs, files in os.walk(path):
        for name in files: 
            if fnmatch.fnmatch(name,pattern): 
                result.append(os.path.join(root,name))
    return result

def find_iter(pattern,path_list):
    return_files = []
    for subpath in path_list:
        dflist = list(Path(subpath).rglob(pattern))
        dflist = [n.as_posix() for n in dflist]
        return_files.append(dflist)
    return_files = list(itertools.chain.from_iterable(return_files))
                      
    return return_files


def json_to_df(path,side_pixpercm,top_pixpercm):

    ### loads experiment metadata into pandas dataframe, currently hardcoded for fps and pixpercm
    data = pd.DataFrame()
    json_files = find('*.txt',path)
    exptfiles = []
    for file in tqdm(json_files):
        with open(file) as j:
            try:
                trial_data = json.load(j)
                df = pd.DataFrame(trial_data['trial_info'])
                ntrials =df.shape[0]
                df['trial'] = np.arange(ntrials).astype(int)+1
                df['fps'] = 60*np.ones(ntrials,).astype('int')
                # df['pixpercm'] = 13*np.ones(ntrials,).astype('int') #measured 13 pix per cm 8/23/20
                df['Top_pixpercm'] = top_pixpercm * np.ones(ntrials,).astype('int') #measured square on jumping platform lip 9/11/20
                df['Side_pixpercm'] = side_pixpercm * np.ones(ntrials,).astype('int') #measured square on jumping platform lip 9/11/20
                if ('suture' in df['condition'].iloc[0]):
                    oc = 'monocular'
                else:
                    oc = 'binocular'
                ocular = [oc] * ntrials
                df['ocular'] = ocular
                df = df.rename(columns={'animal':'subject','laser':'laser_trial'})
                df['laser_trial'] = df['laser_trial'].astype(str)
                df[df['laser_trial']=='0'] = df[df['laser_trial']=='0'].replace('0','laser off')
                df[df['laser_trial']=='1'] = df[df['laser_trial']=='1'].replace('1','laser on')
                # load the corresponding time stamps
                dirname, fname = os.path.split(file)
                ts_df = pd.read_json(os.path.join(dirname,
                    df['expdate'].iloc[0] + '_' + df['subject'].iloc[0] + '_' + df['condition'].iloc[0] + '_' + 'vidclip_ts.txt'))
                new_df = pd.concat([df,ts_df.astype('int')],axis=1)
                data = data.append(new_df,ignore_index=True)
                exptfiles.append(file)
            except:
                pass
    print('%d experiments' % len(exptfiles))
    print('%d trials' % data.shape[0])
    print(data.keys())

    return data, exptfiles

# def csv_to_df(path,side_pixpercm_top_pixpercm):
    


# format starting with eLife revisions, bonsai csv files for trial data
def csv_to_df(path,side_pixpercm,top_pixpercm):
    trial_data_files = find('*TrialData.csv',path)
    grp_columns = ['trial','success','platform','distance','expdate','subject','condition','laser_trial','fps','Top_pixpercm','Side_pixpercm','ocular']
    exp_columns = ['trial','success','platform','distance','laser_trial']
    data = pd.DataFrame(columns=grp_columns)
    exptfiles = []
    for file in trial_data_files:
        
        df = pd.read_csv(file,header=None,names=exp_columns)
        if np.isnan(df['laser_trial'][0]):
            df['laser_trial'] = np.zeros(df.shape[0],dtype=int)

        df['trial'] = df['trial'].astype('int')
        df['success'] = df['success'].astype('int')
        df['platform'] = df['platform'].astype('int')
        df['distance'] = df['distance'].astype('int')
        df['laser_trial'] = df['laser_trial'].astype('int')
        
        df['trial'] +=1

        ntrials =df.shape[0]
        name_list = os.path.split(file)[-1].split('_')
        expdate = [name_list[0]] * ntrials
        subject = [name_list[1]] * ntrials
        condition = [name_list[2]] * ntrials
        df['expdate'] = expdate
        df['subject'] = subject
        df['condition'] = condition
        df['fps'] = 99.97*np.ones(ntrials,)
        df['Top_pixpercm'] = top_pixpercm * np.ones(ntrials,).astype('int') #measured square on jumping platform lip 2/9/22
        df['Side_pixpercm'] = side_pixpercm * np.ones(ntrials,).astype('int') #measured square on jumping platform lip 2/9/22

        if ('suture' in df['condition'].iloc[0]):
            oc = 'monocular'
        else:
            oc = 'binocular'
        ocular = [oc] * ntrials
        df['ocular'] = ocular

        ### add in annotated time stamps for trial events
        print('loading %s' % (df['expdate'].iloc[0] + '_' + df['subject'].iloc[0] + '_' + df['condition'].iloc[0]))
        vidclip_file = find(df['expdate'].iloc[0] + '_' + df['subject'].iloc[0] + '_' + df['condition'].iloc[0] + '*vidclip_ts.txt',path)[0]
        ts_df = pd.read_json(vidclip_file)
        new_df = pd.concat([df,ts_df.astype('int')],axis=1)

        data = data.append(new_df,ignore_index=True)
        exptfiles.append(file)
    return data, exptfiles



def load_dlc_h5(data,cams,dlc_path,likelihood):

    ### loads dlc data for teach trial in the 'data' dataframe based on cams (e.g. 'Side') in the path dlc_path

    df = data.copy()
    row = df.iloc[0]
    fname = find(str(row['expdate']) + '_' + row['subject'] + '_' + cams[0] + '_' + str(row['trial']).zfill(3) + '*filtered.h5',dlc_path)[0]
    pts = pd.read_hdf(fname)
    pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]
    keys = pts.keys()#[:12]
    # keys = ['Nose','LEye','LEar'] #only grab the three head points
    a = [np.zeros((1)) for i in range(df.shape[0])]
    # a = np.zeros((df.shape[0]))
    # a[:] = np.nan
    for key in keys:
        for side in cams:
            df[side + ' ' + key] = a#.astype(object)
    df['jumpdist'] = np.zeros((df.shape[0]))
    # df = df.astype(object)
    # box_sz = 5 #for box filtering
    # box = np.ones(box_sz)/box_sz

    for index, row in tqdm(df.iterrows()):
        for side in cams:
            try:
                fname = find(str(row['expdate']) + '_' + row['subject'] + '_' + side + '_' + str(row['trial']).zfill(3) + '*filtered.h5',dlc_path)[0]
            except:
                print('there is no file for: ', row)
                break
            pts = pd.read_hdf(fname)
            pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]

            for key in keys:
                trace = pts[key].copy().to_numpy()# .iloc[0:row['%s_Jump' % side]-row['%s_Start' % side]].to_numpy()
                if (key=='LEar x') & (side=='Top'):
                        df['jumpdist'].iloc[index] = (trace[row['Top_Jump']-row['Top_Start']] - \
                            trace[row['Top_End']-row['Top_Start']])/row['Top_pixpercm']
                # if (' x' in key) | (' y' in key):
                #     like = pts[key[:-1] + 'likelihood'] #grab likelihoods
                #     trace[like<likelihood] = np.nan #nan out low likelihoods
                #     trace = pd.Series(trace).interpolate().to_numpy() #interp over nans
                #     trace = pd.Series(trace).fillna(method='bfill').to_numpy() #if first vals are nans fill them
                #     trace = signal.medfilt(trace,kernel_size=3) #mean filter
                #     trace = np.convolve(trace, box, mode='same') #smoothing filter
                #     if ' y' in key:
                #         trace = -(trace-np.mean(trace))+np.mean(trace) #flip y trace
                df[side + ' ' + key].iloc[index] = trace.copy()
    return df



def load_dlc_h5_decision_period_only(data,cams,dlc_path,likelihood,filtered):

    ### loads dlc data for teach trial in the 'data' dataframe based on cams (e.g. 'Side') in the path dlc_path

    df = data.copy()
    row = df.iloc[0]
    if filtered:
        try:
            fname = find(str(row['expdate']) + '_' + row['subject'] + '_' + cams[0] + '_' + str(row['trial']).zfill(3) + '*filtered.h5',dlc_path)[0]
        except:
            fname = find_iter(str(row['expdate']) + '*' + row['subject'] + '*' + str(row['trial']).zfill(3) + '*filtered.h5',dlc_path)[0]
    else:
        try:
            fname = find(str(row['expdate']) + '_' + row['subject'] + '_' + cams[0] + '_' + str(row['trial']).zfill(3) + '*.h5',dlc_path)[0]
        except:
            fname = find(str(row['expdate']) + '_' + row['subject'] + '_' + str(row['trial']).zfill(3) + '_' + cams[0] + '*.h5',dlc_path)[0]
    pts = pd.read_hdf(fname)
    pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]
    keys = pts.keys()
    a = [np.zeros((1)) for i in range(df.shape[0])]
    # a = np.empty((df.shape[0]))
    # a[:] = np.nan
    for key in keys:
        for side in cams:
            df[side + ' ' + key] = a.copy()#.astype(object)
    df['jumpdist'] = np.zeros((df.shape[0]))
    # df = df.astype(object)
    box_sz = 5 #for box filtering
    box = np.ones(box_sz)/box_sz

    for index, row in df.iterrows():
        print('trying %s' % (row['expdate'] + '_' + row['subject'] + '_' + row['condition'] + '_' + str(row['trial'])))
        for side in cams:
            if filtered:
                try:
                    fname = find(str(row['expdate']) + '_' + row['subject'] + '_' + cams[0] + '_' + str(row['trial']).zfill(3) + '*filtered.h5',dlc_path)[0]
                except:
                    fname = find_iter(str(row['expdate']) + '*' + row['subject'] + '*' + str(row['trial']).zfill(3) + '*filtered.h5',dlc_path)[0]
            else:
                try:
                    fname = find(str(row['expdate']) + '_' + row['subject'] + '_' + cams[0] + '_' + str(row['trial']).zfill(3) + '*.h5',dlc_path)[0]
                except:
                    fname = find(str(row['expdate']) + '_' + row['subject'] + '_' + str(row['trial']).zfill(3) + '_' + cams[0] + '*.h5',dlc_path)[0]
            # print('doing %s' % fname)
            pts = pd.read_hdf(fname)
            pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]

            for key in keys:
                trace = pts[key].copy().to_numpy()
                if (' x' in key) | (' y' in key):
                    if (key=='LEar x') & (side=='Top'):
                        df['jumpdist'].iloc[index] = (trace[row['Top_Jump']-row['Top_Start']] - \
                            trace[row['Top_End']-row['Top_Start']])/row['Top_pixpercm']
                    trace = trace[:row['%s_Jump' % side]-row['%s_Start' % side]]
                    like = pts[key[:-1] + 'likelihood'] #grab likelihoods
                    like = like[:row['%s_Jump' % side]-row['%s_Start' % side]]
                    trace[like<likelihood] = np.nan #nan out low likelihoods
                    trace = pd.Series(trace).interpolate().to_numpy() #interp over nans
                    trace = pd.Series(trace).fillna(method='bfill').to_numpy() #if first vals are nans fill them
                    trace = signal.medfilt(trace,kernel_size=3) #mean filter
                    trace = np.convolve(trace, box, mode='same') #smoothing filter
                    if ' y' in key:
                        trace = -(trace-np.mean(trace))+np.mean(trace) #flip y trace
                else:
                    trace = trace[:row['%s_Jump' % side]-row['%s_Start' % side]]
                df[side + ' ' + key].iloc[index] = trace.copy().astype(float)

    return df


def load_dlc_h5_jump_data(data,cams,dlc_path,likelihood):

    ### loads dlc data for teach trial in the 'data' dataframe based on cams (e.g. 'Side') in the path dlc_path

    df = data.copy()
    row = df.iloc[0]
    try:
        fname = find(row['expdate'] + '_' + row['subject'] + '_' + row['condition'] + '_' + str(row['trial']-1) + '_' + cams[0] + '*filtered.h5',dlc_path)[0] #trial is 0 indexed in file, 1 in df
    except:
        fname = find(row['expdate'] + '_' + row['subject'] + '_' + row['condition'] + '_' + cams[0] + '_' + str(row['trial']-1) + 'DLC*filtered.h5',dlc_path)[0]
    pts = pd.read_hdf(fname)
    pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]
    keys = pts.keys()
    # keys = ['LEar x', 'LEar y', 'LEar likelihood','TakeFL x','TakeFL y','TakeFL likelihood','TakeFR x','TakeFR y','TakeFR likelihood',\
    # 'LandFL x','LandFL y','LandFL likelihood','LandFR x','LandFR y','LandFR likelihood']
    a = [np.zeros((1)) for i in range(df.shape[0])]
    for key in keys:
        for side in cams:
            df[side + ' ' + key] = a.copy()#.astype(object)
    df['jumpdist'] = np.zeros((df.shape[0]))

    for index, row in df.iterrows():
        for side in cams:
            print('trying %s' % (row['expdate'] + '_' + row['subject'] + '_' + row['condition'] + '_' + str(row['trial']-1)))
            try:
                fname = find(row['expdate'] + '_' + row['subject'] + '_' + row['condition'] + '_' + str(row['trial']-1) + '_' + side + '*filtered.h5',dlc_path)[0] #trial is 0 indexed in file, 1 in df
            except:
                fname = find(row['expdate'] + '_' + row['subject'] + '_' + row['condition'] + '_' + side + '_' + str(row['trial']-1) + 'DLC*filtered.h5',dlc_path)[0]
            # print('doing %s' % fname)
            pts = pd.read_hdf(fname)
            pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]

            for key in keys:
                trace = pts[key].copy().to_numpy()
                if (' x' in key) | (' y' in key):
                    if (key=='LEar x') & (side=='Top'):
                        df['jumpdist'].iloc[index] = (trace[int(row['Top_Jump'])] - trace[int(row['Top_End'])])/row['Top_pixpercm']
                    trace = trace[int(row['%s_Start' % side]):int(row['%s_End' % side])]
                    like = pts[key[:-1] + 'likelihood'] #grab likelihoods
                    like = like[int(row['%s_Start' % side]):int(row['%s_End' % side])]
                    trace[like<likelihood] = np.nan #nan out low likelihoods
                    trace = pd.Series(trace).interpolate().to_numpy() #interp over nans
                    trace = pd.Series(trace).fillna(method='bfill').to_numpy() #if first vals are nans fill them
                    # trace = signal.medfilt(trace,kernel_size=3) #mean filter
                    # trace = np.convolve(trace, box, mode='same') #smoothing filter
                    # if ' y' in key:
                    #     trace = -(trace-np.mean(trace))+np.mean(trace) #flip y trace
                else:
                    trace = trace[int(row['%s_Start' % side]):int(row['%s_End' % side])]
                df[side + ' ' + key].iloc[index] = trace.copy().astype(np.float32)

    return df


def load_dlc_h5_decision_period_only_revisions(data,cams,dlc_path,likelihood):

    ### loads dlc data for teach trial in the 'data' dataframe based on cams (e.g. 'Side') in the path dlc_path

    df = data.copy()
    row = df.iloc[0]
    try:
        fname = find(row['expdate'] + '_' + row['subject'] + '_' + row['condition'] + '_' + str(row['trial']-1) + '_' + cams[0] + '*filtered.h5',dlc_path)[0] #trial is 0 indexed in file, 1 in df
    except:
        fname = find(row['expdate'] + '_' + row['subject'] + '_' + row['condition'] + '_' + cams[0] + '_' + str(row['trial']-1) + 'DLC*filtered.h5',dlc_path)[0]    
    pts = pd.read_hdf(fname)
    pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]
    keys = pts.keys()
    a = [np.zeros((1)) for i in range(df.shape[0])]
    # a = np.empty((df.shape[0]))
    # a[:] = np.nan
    for key in keys:
        for side in cams:
            df[side + ' ' + key] = a.copy()#.astype(object)
    df['jumpdist'] = np.zeros((df.shape[0]))
    # df = df.astype(object)
    box_sz = 5 #for box filtering
    box = np.ones(box_sz)/box_sz

    for index, row in df.iterrows():
        print('trying %s' % (row['expdate'] + '_' + row['subject'] + '_' + row['condition'] + '_' + str(row['trial']-1)))
        for side in cams:
            try:
                fname = find(row['expdate'] + '_' + row['subject'] + '_' + row['condition'] + '_' + str(row['trial']-1) + '_' + side + '*filtered.h5',dlc_path)[0] #trial is 0 indexed in file, 1 in df
            except:
                fname = find(row['expdate'] + '_' + row['subject'] + '_' + row['condition'] + '_' + side + '_' + str(row['trial']-1) + 'DLC*filtered.h5',dlc_path)[0]
            # print('doing %s' % fname)
            pts = pd.read_hdf(fname)
            pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]

            for key in keys:
                trace = pts[key].copy().to_numpy()
                if (' x' in key) | (' y' in key):
                    if (key=='LEar x') & (side=='Top'):
                        df['jumpdist'].iloc[index] = (trace[int(row['Top_Jump'])] - trace[int(row['Top_End'])])/row['Top_pixpercm']
                    trace = trace[int(row['%s_Start' % side]):int(row['%s_Jump' % side])]
                    like = pts[key[:-1] + 'likelihood'] #grab likelihoods
                    like = like[int(row['%s_Start' % side]):int(row['%s_Jump' % side])]
                    trace[like<likelihood] = np.nan #nan out low likelihoods
                    trace = pd.Series(trace).interpolate().to_numpy() #interp over nans
                    trace = pd.Series(trace).fillna(method='bfill').to_numpy() #if first vals are nans fill them
                    trace = signal.medfilt(trace,kernel_size=3) #mean filter
                    trace = np.convolve(trace, box, mode='same') #smoothing filter
                    if ' y' in key:
                        trace = -(trace-np.mean(trace))+np.mean(trace) #flip y trace
                else:
                    trace = trace[int(row['%s_Start' % side]):int(row['%s_Jump' % side])]
                df[side + ' ' + key].iloc[index] = trace.copy().astype(np.float32)

    return df



def load_dlc_h5_decision_period_only_singlecam(data,cam,dlc_path,likelihood,filtered):

    ### loads dlc data for teach trial in the 'data' dataframe based on cams (e.g. 'Side') in the path dlc_path

    df = data.copy()
    row = df.iloc[0]
    if filtered:
        fname = find_iter(str(row['expdate']) + '*' + row['subject'] + '*' + '_' + str(row['trial']).zfill(3) + '*filtered.h5',dlc_path)[0]
    else:
        fname = find_iter(str(row['expdate']) + '*' + row['subject'] + '*' + '_' + str(row['trial']).zfill(3) + '*.h5',dlc_path)[0]
    pts = pd.read_hdf(fname)
    pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]
    keys = pts.keys()
    a = [np.zeros((1)) for i in range(df.shape[0])]
    # a = np.empty((df.shape[0]))
    # a[:] = np.nan
    for key in keys:
        df[cam + ' ' + key] = a.copy()#.astype(object)
    df['jumpdist'] = np.zeros((df.shape[0]))
    # df = df.astype(object)
    box_sz = 5 #for box filtering
    box = np.ones(box_sz)/box_sz

    for index, row in df.iterrows():
        if filtered:
            fname = find_iter(str(row['expdate']) + '*' + row['subject'] + '*' + '_' + str(row['trial']).zfill(3) + '*filtered.h5',dlc_path)[0]
        else:
            fname = find_iter(str(row['expdate']) + '*' + row['subject'] + '*' + '_' + str(row['trial']).zfill(3) + '*.h5',dlc_path)[0]
        
        pts = pd.read_hdf(fname)
        pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]

        for key in keys:
            trace = pts[key].copy().to_numpy()
            if (' x' in key) | (' y' in key):
                if (key=='LEar x'):
                    df['jumpdist'].iloc[index] = (trace[row['%s_Jump' % cam]-row['%s_Start' % cam]] - \
                        trace[row['%s_End' % cam]-row['%s_Start' % cam]])/row['%s_pixpercm' % cam]
                trace = trace[:row['%s_Jump' % cam]-row['%s_Start' % cam]]
                like = pts[key[:-1] + 'likelihood'] #grab likelihoods
                like = like[:row['%s_Jump' % cam]-row['%s_Start' % cam]]
                trace[like<likelihood] = np.nan #nan out low likelihoods
                trace = pd.Series(trace).interpolate().to_numpy() #interp over nans
                trace = pd.Series(trace).fillna(method='bfill').to_numpy() #if first vals are nans fill them
                trace = signal.medfilt(trace,kernel_size=3) #mean filter
                trace = np.convolve(trace, box, mode='same') #smoothing filter
                if ' y' in key:
                    trace = -(trace-np.mean(trace))+np.mean(trace) #flip y trace
            else:
                trace = trace[:row['%s_Jump' % cam]-row['%s_Start' % cam]]
            df[cam + ' ' + key].iloc[index] = trace.copy()

    return df

def find_first(item, vec):
    return np.argmin(np.abs(vec-item))

def flatten_list(arr):
    flat_list = []
    for sublist in arr:
        for item in sublist:
            flat_list.append(item)
    return flat_list

def filter_trace(trace):
    trace = trace[2:-2]
    # mn = np.mean(trace)
    # trace -= mn
    # trace = signal.medfilt(trace) #default is 3
    # trace = np.convolve(trace,np.ones((5,))/5,'same')
    # trace+=mn
    
    return trace

def add_movements_to_df(data,pwin,cams,metric):
    df = data.copy()
    a = [[] for i in range(df.shape[0])]
    # a = np.empty((df.shape[0]))
    # a[:] = np.nan
    keys = ['eye_x_mvmnt', 'eye_y_mvmnt', 'nose_x_mvmnt', 'nose_y_mvmnt',
            'mov_rel_jump', 'mov_rel_start', 'windows']
    for key in keys:
        for cam in cams:
            df[cam + '_' + key] = a
    # df = df.astype(object)
    
    for index, row in tqdm(df.iterrows()):
        for cam in cams:
            ppcm = row[cam + '_pixpercm']
            bobstart = 0 #bob period starts at beginning of clipped trace
            # bobend = row['%s_Jump' % cam]-row['%s_Start' % cam]
            bobend = -1 #revisions
            eye_x = row[cam + ' LEye x'][bobstart:bobend].copy()
            eye_y = row[cam + ' LEye y'][bobstart:bobend].copy()
            # eye_y = -(eye_y-np.mean(eye_y))+np.mean(eye_y)
            nose_x = row[cam + ' Nose x'][bobstart:bobend].copy()
            nose_y = row[cam + ' Nose y'][bobstart:bobend].copy()
            # nose_y = -(nose_y-np.mean(nose_y))+np.mean(nose_y)

            eye_x_mvmnt, eye_y_mvmnt, nose_x_mvmnt, nose_y_mvmnt, mov_rel_jump, mov_rel_start, windows = \
            find_movements(filter_trace(eye_x)/ppcm, filter_trace(eye_y)/ppcm,
            filter_trace(nose_x)/ppcm, filter_trace(nose_y)/ppcm, row['fps'],pwin,metric)
            
            df[cam + '_eye_x_mvmnt'].iloc[index] = eye_x_mvmnt
            df[cam + '_eye_y_mvmnt'].iloc[index] = eye_y_mvmnt
            df[cam + '_nose_x_mvmnt'].iloc[index] = nose_x_mvmnt
            df[cam + '_nose_y_mvmnt'].iloc[index] = nose_y_mvmnt
            df[cam + '_mov_rel_jump'].iloc[index] = mov_rel_jump
            df[cam + '_mov_rel_start'].iloc[index] = mov_rel_start
            df[cam + '_windows'].iloc[index] = np.array(windows)
    return df

def find_movements(eye_x,eye_y,nose_x,nose_y,fps,pwin,metric):
    # pwin = int(pwin*fps) #half window in frames
    dcutoff = 6 # cm total movement cutoff for each dimension
    lcutoff = 0.5 # cm minimum movement requirement
    # vcutoff = 20*(1000/fps) #20cm/s velocity cutoff for successive frames
    
    #get all the windows based on eye trace
    if metric=='velocity':
        windows = get_all_windows_yvelocity(eye_y,pwin,fps)
    elif metric=='position':
        windows,prominences = get_all_windows(eye_x,eye_y,pwin,fps)
        # ditch windows that go outside the trace boundaries
        starts = windows[0,:]>0
        ends = windows[1,:]<len(eye_x)
        good_pts = starts & ends
        windows = windows[:,good_pts]
        prominences = prominences[good_pts]
    else:
        print('not a valid metric!')
        exit()


    #find the unique windows
    try:
        # get rid of movements that occur within half a window of another one
        if windows.shape[1]>1:
            good_wins = []
            win_st = windows[0,:].copy()
            for w in range(len(win_st)):
                if w==0:
                    good_wins.append(w)
                else:
                    if (win_st[w]-win_st[good_wins[-1]])>(int(pwin*fps)):
                        good_wins.append(w)
            good_wins = np.array(good_wins).astype('int')
            windows = windows[:,good_wins]

        # windows = np.sort(windows,axis=1)
        # windows = windows[:,np.hstack((True,np.diff(windows[0,:])>((pwin*fps)/4)))] # set a minimum time between movements
        
        eye_x_mvmnt = [eye_x[j] for j in [np.arange(windows[0,i],windows[1,i]) for i in range(len(windows[0,:]))]]
        eye_y_mvmnt = [eye_y[j] for j in [np.arange(windows[0,i],windows[1,i]) for i in range(len(windows[0,:]))]]
        nose_x_mvmnt = [nose_x[j] for j in [np.arange(windows[0,i],windows[1,i]) for i in range(len(windows[0,:]))]]
        nose_y_mvmnt = [nose_y[j] for j in [np.arange(windows[0,i],windows[1,i]) for i in range(len(windows[0,:]))]]
        
        if fps==50:
            eye_x_mvmnt = [x[::2] for x in eye_x_mvmnt]
            eye_y_mvmnt = [x[::2] for x in eye_y_mvmnt]
            nose_x_mvmnt = [x[::2] for x in nose_x_mvmnt]
            nose_y_mvmnt = [x[::2] for x in nose_y_mvmnt]
        elif fps==75:
            eye_x_mvmnt = [x[::3] for x in eye_x_mvmnt]
            eye_y_mvmnt = [x[::3] for x in eye_y_mvmnt]
            nose_x_mvmnt = [x[::3] for x in nose_x_mvmnt]
            nose_y_mvmnt = [x[::3] for x in nose_y_mvmnt]
        # good_pts = np.ones(prominences.shape,dtype='bool').tolist()

        # x_good = np.array([(np.max(x)-np.min(x))<dcutoff for x in eye_x_mvmnt])
        # y_good = np.array([(np.max(y)-np.min(y))<dcutoff for y in eye_y_mvmnt])
        # good_pts = ((prominences<dcutoff)&(prominences>lcutoff)).tolist()# (x_good & y_good).tolist()

        ### x or y has to be at least threshold
        good_pts_x = np.array([(np.max(x)-np.min(x))>lcutoff for x in eye_x_mvmnt])
        good_pts_y = np.array([(np.max(y)-np.min(y))>lcutoff for y in eye_y_mvmnt])
        good_pts = (good_pts_x|good_pts_y).tolist()

        ### only y has to reach threshold
        # good_pts = np.array([(np.max(y)-np.min(y))>lcutoff for y in eye_y_mvmnt]).tolist()

        
#         x_good = [np.max(np.gradient(x))<vcutoff for x in eye_x_mvmnt]
#         y_good = [np.max(np.gradient(y))<vcutoff for y in eye_y_mvmnt]
#         good_pts_v = x_good and y_good
        
        # good_pts = good_pts_d# and good_pts_v
        windows = list(compress(windows[0,:],good_pts))
        # windows = windows[0,:]
        
        eye_x_mvmnt = list(compress(eye_x_mvmnt, good_pts))
        eye_y_mvmnt = list(compress(eye_y_mvmnt, good_pts))
        nose_x_mvmnt = list(compress(nose_x_mvmnt, good_pts))
        nose_y_mvmnt = list(compress(nose_y_mvmnt, good_pts))
        
        # eye_x_mvmnt = [np.interp(np.arange(0, len(xtr), len(xtr)/40), np.arange(0, len(xtr)), xtr) for xtr in eye_x_mvmnt]
        # eye_y_mvmnt = [np.interp(np.arange(0, len(xtr), len(xtr)/40), np.arange(0, len(xtr)), xtr) for xtr in eye_y_mvmnt]
        # nose_x_mvmnt = [np.interp(np.arange(0, len(xtr), len(xtr)/40), np.arange(0, len(xtr)), xtr) for xtr in nose_x_mvmnt]
        # nose_y_mvmnt = [np.interp(np.arange(0, len(xtr), len(xtr)/40), np.arange(0, len(xtr)), xtr) for xtr in nose_y_mvmnt]

        
        mov_rel_jump = (np.array(windows) - len(eye_x))/fps
        mov_rel_start = (np.array(windows)/len(eye_x))
    except:
        eye_x_mvmnt = np.array([])
        eye_y_mvmnt = np.array([])
        nose_x_mvmnt = np.array([])
        nose_y_mvmnt = np.array([])
        mov_rel_jump = np.array([])
        mov_rel_start = np.array([])
        windows = np.array([])

    return eye_x_mvmnt, eye_y_mvmnt, nose_x_mvmnt, nose_y_mvmnt, mov_rel_jump, mov_rel_start, windows
    
def get_all_windows(xtrace,ytrace,pwin,fps):
    #parameters for peak finding and window duration
    peakthresh = 0.125 #how to set this now? usually 0.5 for raw traces, but for x^2+y^2?
    distance = int(fps/5)
    ###find peaks from a positive convolution of the data
    # trace = np.square(xtrace) + np.square(ytrace)
    trace = np.sqrt(np.square(xtrace) + np.square(ytrace))
    # trace = np.diff(trace,append=0)
    trace = trace - np.mean(trace)
    peaktimes,props = signal.find_peaks(trace, prominence=peakthresh, distance=distance)
    # peaktimes_neg = signal.find_peaks(-trace, prominence=peakthresh, distance=distance)[0]
    # peaktimes = np.hstack((peaktimes_pos,peaktimes_neg))
    windows = np.vstack((peaktimes-int(pwin*fps),peaktimes + int(pwin*fps)))
    prominences = props['prominences']

    return windows,prominences

def get_all_windows_yvelocity(ytrace,pwin,fps):
    neg_peaks = np.where((np.diff(ytrace[:-1])<0)&(np.diff(ytrace[1:])>0))[0]
    pos_peaks = np.where((np.diff(ytrace[:-1])>0)&(np.diff(ytrace[1:])<0))[0]
    peaktimes = np.sort(np.concatenate((neg_peaks,pos_peaks)))
    peaktimes = peaktimes[(peaktimes>int(pwin*fps))&(peaktimes<(len(ytrace)-int(pwin*fps)))]
    windows = np.vstack((peaktimes-int(pwin*fps),peaktimes + int(pwin*fps)))

    return windows

def plot_movs_per_sec(ax,df_cond,labels_cond,condition,cluster_key,ymin,ymax,color_scheme,save_pdf,pp,suptitle=''):

    anis = np.unique(df_cond['subject'])
    conds = np.unique(df_cond[condition])
    n_hclust = len(np.unique(labels_cond[cluster_key]))
    freq_hclust = np.zeros((n_hclust,len(conds),len(anis)))
    # fig, axs = plt.subplots(1,1,figsize=(2,2))

    for c,cond in enumerate(conds):
        for a,ani in enumerate(anis):
            all_mov = labels_cond[cluster_key][(labels_cond['subject']==ani)&(labels_cond[condition]==cond)].to_numpy()
            for i in np.arange(n_hclust):
                freq_hclust[i,c,a] = sum(all_mov==i)/df_cond['trial_length'][(df_cond['subject']==ani)&(df_cond[condition]==cond)].sum()

        ax.errorbar(x=np.arange(n_hclust) +1 + c*0.2,y=np.nanmean(freq_hclust[:,c,:],axis=1),yerr=np.nanstd(freq_hclust[:,c,:],axis=1)/np.sqrt(len(anis)),color=color_scheme[c],ls='none',marker='o',markersize=3,label=cond)
    ax.set_ylim(ymin,ymax)
    ax.set_xlim(0,n_hclust+1)
    # axs.set_xticks(np.arange(0,n_hclust+2,2))
    ax.set_xlabel('movement cluster')
    ax.set_ylabel('mvmnts/s')
    ax.legend()
    ax = xy_axis(ax)

    # fig.suptitle(suptitle)
    # fig.tight_layout()

    # if save_pdf:
    #     pp.savefig(fig)
    #     plt.close(fig)

    return ax#fig, axs


def play_fn(x_bob,y_bob,embedding,title):
    
    ### a function to visualize the UMAP data with an interactive html file
    
    ### INPUTS
    ### x_bob and y_bob: the raw x and y traces for each movement
    ### theta: angle between eye and nose
    ### embedding: the UMAP embedding from e.g. reduced data
    ### title: file name of the UMAP html file to save out
    
    ### OUTPUTS
    ### p: the interactive UMAP plot
    
    x = embedding[:,0]
    y = embedding[:,1]
    xs = np.repeat(np.arange(x_bob.shape[1])[np.newaxis,:],embedding.shape[0],axis=0)
    ys_bob = y_bob - y_bob[:,:1]
    xs_bob = x_bob - x_bob[:,:1]
    xp = [xi for xi in xs] # Because Multi-List does not like numpy arrays
    yp_bob = [yi for yi in ys_bob]
    xp_bob = [yi for yi in xs_bob]
    # theta_p = [thetai for thetai in theta]
    
    x_start = xs_bob[:,0]
    x_end = xs_bob[:,-1]
    y_start = ys_bob[:,0]
    y_end = ys_bob[:,-1]
    
    output_file(title)
        
    num = np.arange(embedding.shape[0])
    cmap = Turbo256*int(np.ceil(embedding.shape[0]/255))
    cmap = cmap[:embedding.shape[0]]
    source = ColumnDataSource(data=dict(x=x,y=y,xp=xp,yp_bob=yp_bob,xp_bob=xp_bob,
                                        x_start=x_start, x_end=x_end,
                                        y_start=y_start, y_end=y_end,
                                        color=cmap
                                       )) #theta=theta_p,
    TOOLS = ['box_select','wheel_zoom','pan','tap','box_zoom','reset','save']
    # toolList = ['lasso_select', 'tap', 'reset', 'save']

#     exp_cmap = linear_cmap('y', 'Turbo256', 0, 255)
    
    plot_width = 450
    plot_height = 450
    ymin = -5
    ymax = 5
    left = figure(tools=TOOLS,plot_width=plot_width,plot_height=plot_height,title='Embedding',match_aspect=True)
    c1 = left.circle('x','y',source=source)
    c1.nonselection_glyph = Circle(fill_color='gray',fill_alpha=0.4,
                                   line_color=None)
    c1.selection_glyph = Circle(fill_color='color',line_color=None)
    
    right_t = figure(tools=TOOLS,plot_width=plot_width,plot_height=int(np.round(plot_height/3)),title='x Bobs',y_range=(ymin, ymax)) #
#     right_t.add_layout(Title(text="Frames", align="center"), "below")
    c2 = right_t.multi_line(xs='xp',ys='xp_bob',source=source,line_color='color')
    c2.nonselection_glyph = MultiLine(line_color='gray',line_alpha=0.0)
    c2.selection_glyph = MultiLine(line_color='color')
    
    right_b = figure(tools=TOOLS,plot_width=plot_width,plot_height=int(np.round(plot_height/3)),title='y Bobs',y_range=(ymin, ymax))
#     right_b.add_layout(Title(text="Frames", align="center"), "below")
    c3 = right_b.multi_line(xs='xp',ys='yp_bob',source=source,line_color='color')
    c3.nonselection_glyph = MultiLine(line_color='gray',line_alpha=0.0)
    c3.selection_glyph = MultiLine(line_color='color')
#     p = gridplot([[left, right_t], [, right_b]])

    right_bb = figure(tools=TOOLS,plot_width=plot_width,plot_height=int(np.round(plot_height/3)),title='Theta')#,y_range=(-np.pi/2, np.pi/2))
    right_bb.add_layout(Title(text="Frames", align="center"), "below")
#     c3b = right_bb.multi_line(xs='xp',ys='theta',source=source,line_color='color')
#     c3b.nonselection_glyph = MultiLine(line_color='gray',line_alpha=0.0)
#     c3b.selection_glyph = MultiLine(line_color='color')
# #     p = gridplot([[left, right_t], [, right_b]])

    fTrace = figure(tools=TOOLS,plot_width=plot_width,plot_height=plot_height,title='xy Bobs',match_aspect=True,x_range=(-6,6),y_range=(-6, 6))
    fTrace.add_layout(Title(text="x Position", align="center"), "below")
    fTrace.add_layout(Title(text="y Position", align="center"), "left")
    c4 = fTrace.multi_line(xs='xp_bob',ys='yp_bob',source=source,line_color='color')
    c4a = fTrace.circle('x_start','y_start', source=source, color='green')
    c4b = fTrace.circle('x_end','y_end', source=source, color='red')
    c4.nonselection_glyph = MultiLine(line_color='gray',line_alpha=0.0)
    c4.selection_glyph = MultiLine(line_color='color', line_width=3)
    
    c4a.nonselection_glyph = Circle(line_color='gray',fill_alpha=0.0, line_alpha = 0)
#     c4a.selection_glyph = Circle(line_color=mapper, line_width=3)
    c4b.nonselection_glyph = Circle(line_color='gray',fill_alpha=0.0, line_alpha = 0)
#     c4b.selection_glyph = Circle(line_color=mapper, line_width=3)
    
    p = row([left,column(right_t,right_b,right_bb,sizing_mode='scale_both'),fTrace], sizing_mode='scale_both')
    
    show(p)

    return p


def load_timestamps(type,dir):

    ### loads timestamps from csv file for videos recorded in Bonsai

    ### INPUTS
    ### type: either Bonsai or Flir
    ### dir: video directory

    ### OUTPUTS
    ### cams: list of the names of each camera
    ### data: list of numpy arrays containing the actual time stamps

    file_list = find('*' + type + '*.csv',dir)
    cams = []
    data = []
    for file in file_list:
        cams.append(os.path.split(file)[1].split('_')[-2])
        data.append(np.genfromtxt(file))

    return cams,data


def xy_axis(ax):
    
    ### Removes the top and right bounding axes that are plotted by default in matplotlib
    
    ### INPUTS
    ### ax: axis object (e.g. from fig,ax = plt.subplots(1,1))
    
    ### OUTPUTS
    ### ax: the same axis w/top and right lines removed
    
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    return ax


def add_scalebar(ax,x,y,xlab,ylab):
    
    ### Adds scale bar to axis
    
    ### INPUTS
    ### ax: axis object (e.g. from fig,ax = plt.subplots(1,1))
    ### x: the x length of the scale bar (number)
    ### y: the y length of the scale bar (number)
    ### xlab: the label for x scale bar (string)
    ### ylab: the label for y scale bar (string)
    
    ### OUTPUTS
    ### ax: the same axis w/scale bar added
    
    xmax = int(np.floor(ax.get_xlim()[1]))
    ymax = int(np.ceil(ax.get_ylim()[0]))
    
    ax.plot([xmax-x,xmax],[ymax,ymax],'k-')
    ax.plot([xmax,xmax],[ymax,ymax+y],'k-')
    
    ax.text(xmax-x,ymax-0.2,'%s %s' % (x,xlab),fontsize=10)
    ax.text(xmax+0.2,ymax,'%s %s' % (y,ylab),fontsize=10,rotation=90)
    

    return ax



def data_to_umap(df,side):
    
    ### get data into UMAP format
    
    ### INPUTS
    ### df: dataframe containing all the data
    
    ### OUTPUTS
    ### data: the concatenated x/y eye movement velocity data normalized to peak (n_observations,n_time_points)
    ### x and y: the raw x and y eye traces of each movement (800ms in length)
    ### theta: angle between the eye and nose
    ### xnose and ynose: the raw x and y nose traces of each movement (800ms in length)
    ### mov_rel_jump: time of each movement relative to the time of the jump
    ### mov_rel_start: time of each movement normalized to the trial length (between 0 and 1)
    ### windows: original frame number of each movement (can be used to make labeled movies)

    # # Grab Nose Point
    # xnose = df[side + '_nose_x_mvmnt'].values.tolist()
    # ynose = df[side + '_nose_y_mvmnt'].values.tolist()
    # xnose = np.vstack([np.stack(i) for i in xnose if i])
    # ynose = np.vstack([np.stack(i) for i in ynose if i])
    if side=='Side':
        side = 'Side_'
    elif side=='Top':
        side = 'Top_'

    # Grab LEye point
    xeye=df[side + 'eye_x_mvmnt'][:].copy().values.tolist()
    yeye=df[side + 'eye_y_mvmnt'][:].copy().values.tolist()
    xeye = np.vstack([np.stack(i) for i in xeye if i])
    yeye = np.vstack([np.stack(i) for i in yeye if i])

    #zero center eye points
    # ycent=y - y[:,:1]
    # xcent=x - x[:,:1]

    #normalize amplitudes for pca
    xnorm = xeye*(1./np.max(xeye, axis=1)[:,None])
    ynorm = yeye*(1./np.max(yeye, axis=1)[:,None])
    xyeye = np.hstack((xnorm,ynorm))
    data = xyeye
    
    # # get velocity and 0 centered points (not currently doing)
    # dxeye=np.diff(xeye,axis=1)
    # dyeye=np.diff(yeye,axis=1)
    # dycent=dy - dy[:,:1]
    # dxcent=dx - dx[:,:1]

    #mean centered points 
    # dyeyecent = dyeye # - np.mean(dy,axis=1)[:,None]
    # dxeyecent = dxeye # - np.mean(dx,axis=1)[:,None]

    # dxyeye = np.hstack((dxeyecent,dyeyecent)) 
    # dxy = dxy*(1./np.max(dxy, axis=1)[:,None]) #try with normalizing to peak of x/y trace
    # dxy = (dxy.T/dxy.max(axis=1)).T
    # dxy[np.isnan(dxy)] = 0

    # xy = np.hstack((xcent,ycent))

    # not currenly using the nose/eye angle
    # theta =np.unwrap(np.arctan2((ynose-yeye),(xnose-xeye)),np.pi/2)
    # theta = np.degrees(np.unwrap(np.arctan2((y-ynose),(x-xnose)),np.pi/2))
#     dtheta =np.diff(np.unwrap(np.arctan2((ynose-y),(xnose-x)),np.pi/2),axis=1)

    # data = np.hstack((dxy,dtheta))
    # data = dxyeye

    # ynosecent = ynose*(1./np.max(ynose, axis=1)[:,None])
    # xnosecent = xnose*(1./np.max(xnose, axis=1)[:,None])
    # dxnose=np.diff(xnose,axis=1)
    # dynose=np.diff(ynose,axis=1)
    # dynosecent = dynose # - np.mean(dy,axis=1)[:,None]
    # dxnosecent = dxnose # - np.mean(dx,axis=1)[:,None]

    # dxynose = np.hstack((dxnosecent,dynosecent)) 

    # data = np.hstack((dxyeye,dxynose)) 

    # Movement timing
    mov_rel_jump=df[side + 'mov_rel_jump'][:].copy().values.tolist()
    mov_rel_start=df[side + 'mov_rel_start'][:].copy().values.tolist()
    windows=df[side + 'windows'][:].copy().values.tolist()
    mov_rel_jump = np.concatenate([np.array(i) for i in mov_rel_jump])
    mov_rel_start = np.concatenate([np.array(i) for i in mov_rel_start])
    windows = np.concatenate([np.array(i) for i in windows])
    
    print('data shape: ',data.shape)

    return data, xeye, yeye, mov_rel_jump, mov_rel_start, windows #xnose, ynose, theta, 


def run_pca_kmeans(data,clusters,save_pdf,pp):
    
    ### INPUTS
    ### data: raw data in an array with shape (n_observations,n_time_points)
    ### clusters: Either the number of clusters you want (integer) or fraction of total variance you want explained (0 to 1)
    
    ### OUTPUTS
    ### pca: the pca object run on the selected number of components based on 'clusters'
    ### reduced_data: the result of transform fit on data (n_observations,n_components)
    ### kmeans: the kmeans object run on the reduced_data with selected number of components based on 'clusters'
    ### n_k: the number of pca components and kmeans clusters (they're the same)
    
    # first do a pca with the max number of components
    pca = PCA()
    reduced_data = pca.fit_transform(data)
    
    if clusters<1:
        #determine the number of PCs below the desired explained variance
        n_k = np.where(np.cumsum(pca.explained_variance_ratio_)<clusters)[0][-1]
    else:
        n_k = clusters

    fig, ax = plt.subplots(1,1,figsize=(5,5))
    ax.plot(np.cumsum(pca.explained_variance_ratio_),'k-')
    ax.plot([n_k,n_k],[0,2],'r:',label='%s components chosen' % n_k)
    ax.set_ylim(0,1.1)
    ax.set_xlabel('Number of components')
    ax.set_ylabel('Explained variance')
    ax.legend(fontsize=10,loc=4)
    ax = xy_axis(ax)

    if save_pdf:
        pp.savefig(fig)
        plt.close(fig)
    
    #re-run pca with selected number of components
    pca = PCA(n_components=n_k)
    reduced_data = pca.fit_transform(data)

    #perform K-means clustering with same number of clusters as components
    kmeans = KMeans(init='k-means++', n_clusters=n_k, n_init=100) #maybe
    cluster_data = kmeans.fit_transform(reduced_data)

    #Gaussian mixture
    # gmm = mixture.GaussianMixture(n_components=10, covariance_type='full',max_iter=100, n_init=100, init_params='random')
    # gmm_labels = gmm.fit_predict(reduced_data)
    
    return pca, reduced_data, kmeans, cluster_data, n_k, fig, ax


def create_umap_labels(df,side,kmeans,reduced_data,x,y,windows,mov_rel_jump,mov_rel_start):
    
    ### INPUTS
    ### df: dataframe containing all the data
    ### kmeans: the kmeans object from the output of run_pca_kmeans
    ### reduced_data: from PCA analysis, to embed in UMAP
    ### x and y: the eye movement data from the output of data_to_umap
    ### windows: frame numbers of the movement starts
    ### mov_rel_jump: the timing in sec of the movement relative to the jump time
    ### mov_rel_start: the timing in sec of the movement relative to the start of the trial
    
    ### OUTPUTS
    ### labels: a dataframe with one entry for each individual movement
    ### embedding: the PCA data embedded in the UMAP
    if side=='Side':
        side = 'Side_'
    elif side=='Top':
        side = 'Top_'

    #embed the data into the UMAP
    embedding = umap.UMAP(
        n_neighbors=30,
        min_dist=0.0,
        n_components=2,
        random_state=42,
    ).fit_transform(reduced_data) #cluster_data or reduced_data (PCA)

    print('embedded data shape: ',embedding.shape)

    y_full=df[side + 'eye_y_mvmnt'].copy()
    x_full=df[side + 'eye_x_mvmnt'].copy()
    subject = np.hstack([np.hstack([df['subject'].copy()[n]]*len(i)) for n,i in enumerate(y_full) if i])
    expdate = np.hstack([np.hstack([df['expdate'].copy()[n]]*len(i)) for n,i in enumerate(y_full) if i])
    distance = np.hstack([np.hstack([df['distance_DLC'].copy()[n]]*len(i)) for n,i in enumerate(y_full) if i]).astype('float')
    success = np.hstack([np.hstack([df['success'].copy()[n]]*len(i)) for n,i in enumerate(y_full) if i])
    platform = np.hstack([np.hstack([df['platform'].copy()[n]]*len(i)) for n,i in enumerate(y_full) if i])-1
    condition = np.hstack([np.hstack([df['condition'].copy()[n]]*len(i)) for n,i in enumerate(y_full) if i])
    ocular = np.hstack([np.hstack([df['ocular'].copy()[n]]*len(i)) for n,i in enumerate(y_full) if i])
    trial = np.hstack([np.hstack([df['trial'].copy()[n]]*len(i)) for n,i in enumerate(y_full) if i])
    laser_trial = np.hstack([np.hstack([df['laser_trial'].copy()[n]]*len(i)) for n,i in enumerate(y_full) if i])
    jumpdist = np.hstack([np.hstack([df['jumpdist'].copy()[n]]*len(i)) for n,i in enumerate(y_full) if i])
    

    # mov_rel_jump = np.hstack([np.hstack([df['mov_rel_jump'][n]]*len(i)) for n,i in enumerate(y_full) if i])/df['fps'][0]
    # mov_rel_start = np.hstack([np.hstack([df['mov_rel_start'][n]]*len(i)) for n,i in enumerate(y_full) if i])

    labels = pd.DataFrame({'subject': subject,'expdate':expdate,'distance_DLC':distance.astype('float'),
                           'success':success.astype(np.uint16),'platform':platform.astype(np.uint16),'jumpdist':jumpdist,
                           'condition':condition,'ocular':ocular,'trial':trial,'laser_trial':laser_trial,side + 'windows':windows,
                           side + 'mov_rel_jump':mov_rel_jump,side + 'mov_rel_start':mov_rel_start})#, 'xory':xory.astype(np.uint16)})
    labels['subject_id'] = pd.factorize(labels['subject'].copy())[0].astype(np.uint16)
    labels['distance_id'] = pd.factorize(labels['distance_DLC'].copy())[0].astype(np.uint16)
    labels['condition_id'] = pd.factorize(labels['condition'].copy())[0].astype(np.uint16)
    labels['kmeans'] = kmeans.labels_
    labels['embedding_x'] = embedding[:,0]
    labels['embedding_y'] = embedding[:,1]
    ##if doing gaussian mixtures:
    #labels['gmm'] = gmm_labels

    x_eye_mvmnt = [item for item in x]
    y_eye_mvmnt = [item for item in y]
    labels[side + 'x_eye_mvmnt'] = x_eye_mvmnt
    labels[side + 'y_eye_mvmnt'] = y_eye_mvmnt
    
    print('labels shape: ',labels.shape)
    print('labels keys: ',[label for label in labels.keys()])
    
    return labels, embedding


def order_clusters_by_variance(labels,side,key):
    
    ### Takes in labels dataframe and reorders the cluster numbers in key by variance (largest to smallest)
    
    ### INPUTS
    ### labels: a dataframe with one entry for each individual movement
    ### key: the key you want to reorder, e.g. kmeans orcluster_key    
    ### OUTPUTS
    ### labels: with reordered cluster labels
    
    n_k = len(np.unique(labels[key]))
    x = np.vstack(labels[side + '_x_eye_mvmnt'].copy().tolist())
    y = np.vstack(labels[side + '_y_eye_mvmnt'].copy().tolist())
    clust_var = np.array([])
    for label in range(n_k):
        tr_idx = np.where(labels[key]==label)[0]
        xtr = np.mean(x[tr_idx,:],axis=0)
        xtr -= np.mean(xtr)
        ytr = np.mean(y[tr_idx,:],axis=0)
        ytr -= np.mean(ytr)

        clust_var = np.append(clust_var,np.var(np.concatenate((xtr,ytr))))
        
    new_kmeans = np.argsort(-clust_var)
    old_kmeans = np.arange(n_k)
    old_labels = np.array(labels[key].tolist())
    new_labels = np.array(old_labels.copy())

    for ok,nk in zip(old_kmeans,new_kmeans):
        idx = np.where(old_labels==nk)[0]
        new_labels[idx] = ok

    del labels[key]
    labels[key] = new_labels
              
    return labels
    

def make_sequential_list(ls):
    
    ### Used to make the list of colors in the dendogram, right now leaves out blue because it's the main cluster color (all data)
    
    ### INPUTS
    ### ls: a list of colors from the dendogram (dend['color_list'])
    
    ### OUTPUTS
    ### a list of unique colors in the order they appear in the list
    
    out_ls = []
    for l in ls:
        if l in out_ls:
            pass
        elif l=='b':
            pass
        else:
            out_ls.append(l)
    return out_ls


def hierarchical_dendrogram(reduced_data,n_guess):
    Z = shc.linkage(reduced_data, method='ward')
    fig1 = plt.figure(figsize=(7,7))
    dend = shc.dendrogram(Z,p=n_guess,truncate_mode='lastp',distance_sort='ascending',show_leaf_counts=False)
    print('finished calculating linkage!')
    return Z

def hierarchical_clustering(labels,side,reduced_data,Z,cthresh,n_guess,plot_params,save_pdf,pp):
    
    ### Plot a hierarchy of clusters to obtain the number of clusters to feed into agglomerative clustering
    
    ### INPUTS
    ### labels: a dataframe with one entry for each individual movement    
    ### reduced_data: the result of transform fit on data (n_observations,n_components)
    ### cthresh: the threshold for the length at which to cut off clustering (currently using 9)
              
    ### OUTPUTS
    ### labels: the same dataframe but with a new column called 'hclust' containing hierarchical cluster IDs
    if side=='Side':
        side = 'Side_'
    elif side=='Top':
        side = 'Top_'

    cols, col_map = make_colormap(n_guess)
    shc.set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in cols])

    # print('making dendrogram (fig1)')
    fig1 = plt.figure(figsize=(10,10))
    dend = shc.dendrogram(Z,p=n_guess,truncate_mode='lastp',distance_sort='ascending',show_leaf_counts=False,color_threshold=cthresh)
    # print('finished making dendrogram')
    plt.axhline(y=cthresh, color='r', linestyle='--')
    
    if save_pdf:
        pp.savefig(fig1)
        plt.close(fig1)
        # print('saved fig1 to pdf')

    # Plot clusters based off distances metric from left to right of dendrogram
    # print('getting clusters')
    fclusters = shc.fcluster(Z,criterion='distance',t=cthresh)
    nclusters = len(np.unique(fclusters))
    print('%d fclusters' % nclusters)
    # ivl = np.array([int(i) for i in dend['ivl']])
    # print('plotting clusters (fig2)')
    fig2, axs2 = plt.subplots(2,int(np.ceil(nclusters/2)),figsize=(2.5*int(np.ceil(nclusters/2)),5))
    x = np.vstack(labels[side + 'x_eye_mvmnt'].copy().tolist())
    y = np.vstack(labels[side + 'y_eye_mvmnt'].copy().tolist())
    for cl,ax in zip(np.unique(fclusters),axs2.flatten()):
        ivl2 = np.array([n for n,i in enumerate(fclusters) if i ==cl])
        xtr = x[ivl2,:] - np.mean(x[ivl2,:],axis=1)[:,np.newaxis]
        ytr = y[ivl2,:] - np.mean(y[ivl2,:],axis=1)[:,np.newaxis]
        xavg = np.mean(x[ivl2,:],axis=0) - np.mean(x[ivl2,:],axis=(0,1))
        yavg = np.mean(y[ivl2,:],axis=0) - np.mean(y[ivl2,:],axis=(0,1))
        ax.plot(xtr.T,ytr.T,color=[0.7,0.7,0.7],linewidth=1)
        ax.plot(xavg,yavg,c=cols[cl-1],linewidth=3)
        ax.plot(xavg[0],yavg[0],'bo')
        ax.plot(xavg[-1],yavg[-1],'ro')
        ax.axis([-3,3,-3,3])
        ax.axis('off')
    ax = add_scalebar(ax,1,1,'cm','cm')
    
    if save_pdf:
        pp.savefig(fig2)
        plt.close(fig2)
        # print('saved fig2 to pdf')
    
    # print('performing agglomerative clustering')
    cluster = AgglomerativeClustering(n_clusters=nclusters, affinity='euclidean', linkage='ward')  
    cluster.fit_predict(reduced_data)
    # np.unique(cluster.labels_)
    labels_new = labels.copy()
    labels_new['hclusters'] = cluster.labels_
    # print('finished agg clustering')
    
    return labels_new, nclusters, fig1, fig2


def plot_regression(x,y):
    
    ### performs a linear regression on two 1-D input arrays and returns new x and y values to plot a line with
    
    ### INPUTS
    ### x and y: 1-D arrays containing the data to fit
    
    ### OUTPUTS
    ### x_new and y_new: points resulting from linear regression on inputs
    model = LinearRegression()
    model.fit(np.reshape(x,(-1,1)),np.reshape(y,(-1,1)))
    x_new=np.linspace(x[0],x[-1],100)
    y_new = model.predict(x_new[:,np.newaxis])
    
    return x_new,y_new


def get_movement_amplitudes(labels,side):

    ### Adds columns to labels with the x and y amplitudes for each individual movement

    ### INPUTS
    ### labels: the dataframe created by create_umap_labels

    ### OUTPUTS
    ### labels: the same dataframe with two new columns appended, xamps and yamps

    #peak finding parameters
    prominence=0.1
    pdist=5

    xamps,yamps,xvel,yvel,xamps_idx,yamps_idx,xvel_idx,yvel_idx,visangle = ([] for i in range(9))
    for index,row in labels.iterrows():

        xtr = row[side + '_x_eye_mvmnt'].copy()
        ytr = row[side + '_y_eye_mvmnt'].copy()
        dxtr = np.diff(xtr)
        dytr = np.diff(ytr)

        xpt,xptp = signal.find_peaks(xtr, prominence=prominence, distance=pdist)
        ypt,yptp = signal.find_peaks(ytr, prominence=prominence, distance=pdist)
        dxpt,dxptp = signal.find_peaks(dxtr, prominence=prominence, distance=pdist)
        dypt,dyptp = signal.find_peaks(dytr, prominence=prominence, distance=pdist)

        if (len(xpt)==0) and (len(ypt)==0):
            xamps.append(np.nan)
            yamps.append(np.nan)
            xvel.append(np.nan)
            yvel.append(np.nan)
            xamps_idx.append(np.nan)
            yamps_idx.append(np.nan)
            xvel_idx.append(np.nan)
            yvel_idx.append(np.nan)
        elif len(xpt)==0:
            idx = np.argmin(abs(ypt-len(ytr)/2).astype('int'))
            yamps.append(yptp['prominences'].copy()[idx])
            yamps_idx.append(ypt[idx])
            try:
                yvel.append(dyptp['prominences'].copy()[idx])
                yvel_idx.append(dypt[idx])
            except:
                yvel.append(np.max(dytr))
                yvel_idx.append(np.argmax(dytr))
            L = yptp['left_bases'].copy()[idx]
            R = yptp['right_bases'].copy()[idx]
            xamps.append(np.abs(xtr[L]-xtr[R]))
            xamps_idx.append(np.argmax(xtr))
            xvel.append(np.max(dxtr[L:R]))
            xvel_idx.append(np.argmax(dxtr[L:R]))
        elif len(ypt)==0:
            idx = np.argmin(abs(xpt-len(xtr)/2).astype('int'))
            xamps.append(xptp['prominences'].copy()[idx])
            xamps_idx.append(xpt[idx])
            try:
                xvel.append(dxptp['prominences'].copy()[idx])
                xvel_idx.append(dxpt[idx])
            except:
                xvel.append(np.max(dxtr))
                xvel_idx.append(np.argmax(dxtr))
            L = xptp['left_bases'].copy()[idx]
            R = xptp['right_bases'].copy()[idx]
            yamps.append(np.abs(ytr[L]-ytr[R]))
            yamps_idx.append(np.argmax(ytr))
            yvel.append(np.max(dytr[L:R]))
            yvel_idx.append(np.argmax(dytr[L:R]))
        else:
            idx = np.argmin(abs(xpt-len(xtr)/2).astype('int'))
            xamps.append(xptp['prominences'].copy()[idx])
            xamps_idx.append(xpt[idx])
            try:
                xvel.append(dxptp['prominences'].copy()[idx])
                xvel_idx.append(dxpt[idx])
            except:
                xvel.append(np.max(dxtr))
                xvel_idx.append(np.argmax(dxtr))
            idx = np.argmin(abs(ypt-len(ytr)/2).astype('int'))
            yamps.append(yptp['prominences'].copy()[idx])
            yamps_idx.append(ypt[idx])
            try:
                yvel.append(dyptp['prominences'].copy()[idx])
                yvel_idx.append(dypt[idx])
            except:
                yvel.append(np.max(dytr))
                yvel_idx.append(np.argmax(dytr))

        dist = row['distance_DLC']
        visangle.append(np.round(np.degrees(np.arctan(yamps[-1]/dist)),1))


    labels['xamps'] = xamps
    labels['yamps'] = yamps
    labels['xvel'] = xvel
    labels['yvel'] = yvel
    labels['xamps_idx'] = xamps_idx
    labels['yamps_idx'] = yamps_idx
    labels['xvel_idx'] = xvel_idx
    labels['yvel_idx'] = yvel_idx
    labels['visangle'] = visangle

    return labels


def make_colormap(n):
    cols = cm.jet(np.linspace(0, 1, n))
    col_map = ListedColormap(cols)
    return cols, col_map

def plot_UMAP_labels(labels,key,plot_params,save_pdf,pp):
    
    ### INPUTS
    ### labels: the dataframe created by create_umap_labels
    ### key: the key in labels that you want to plot (e.g. subject, condition)
    ### save_pdf: whether option was selected to save pdf
    ### pp: the pdf object (only necessary if saving PDF)
    
    ### OUTPUTS
    ### fig, ax: the figure and axes objects
    n = len(np.unique(labels[key]))
    cols, col_map = make_colormap(n)
#     cnorm = BoundaryNorm(boundaries=np.arange(n),ncolors=n)

    
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    try:
        pl1 = ax.scatter(labels['embedding_x'],labels['embedding_y'],c=labels[key],
                         cmap=col_map,s=plot_params['pointsize']) #norm=cnorm,
    except:
        pl1 = ax.scatter(labels['embedding_x'],labels['embedding_y'],c=pd.factorize(labels[key])[0].astype(np.uint16),
                         cmap=col_map,s=plot_params['pointsize']) #norm=cnorm,
    ax.set_title(key)
    ax.set_aspect('equal', 'datalim')
    ax.axis('off')
    ax = add_scalebar(ax,1,1,'a.u.','a.u.')
    plt.colorbar(pl1,ax=ax,
                 boundaries=np.arange(len(np.unique(labels[key]))+1)-0.5).set_ticks(np.arange(len(np.unique(labels[key]))))

    # plt.show()

    if save_pdf:
        pp.savefig(fig)
        plt.close(fig)
        
    return fig, ax


def plot_cluster_freq(ax,df,labels,cluster_key,condition,y_min,y_max,color_scheme,save_pdf,pp):
    
    ### Plot frequency of each cluster averaged across animals
    
    ### INPUTS
    ### df: the dataframe with all the data
    ### labels: the dataframe with the UMAP labels from create_umap_labels
    ### condition: the key from df that you want to split the data into (e.g. 'ocular')
    ### plot_params: a dictionary of parameters including a colormap 'cm' and colors 'cond_cols'
    ### save_pdf: whether option was selected to save pdf
    ### pp: the pdf object (only necessary if saving PDF)
    
    ### OUTPUTS
    ### fig, ax: the figure and axes objects
    
    n_k = len(np.unique(labels[cluster_key]))
    conds = np.unique(df[condition])
    cols, col_map = make_colormap(n_k)
    anis = np.unique(df['subject'])

    df = df[df['success']==1]
    df.reset_index(inplace=True,drop=True)
    labels = labels[labels['success']==1]
    labels.reset_index(inplace=True,drop=True)
    
    #num trials in each cond
    cond_trials = np.zeros((len(conds),len(anis)))
    for c,cond in enumerate(conds):
        for a,ani in enumerate(anis):
            cond_trials[c,a] = df[(df[condition]==cond)&(df['subject']==ani)].shape[0]
            
    freq = np.zeros((len(conds),len(anis),n_k))
    for c,cond in enumerate(conds):
        for label in range(n_k):
            for a,ani in enumerate(anis):
                tr_idx = np.intersect1d(np.intersect1d(np.where(labels[cluster_key]==label)[0],np.where(labels[condition]==cond)[0]),np.where(labels['subject']==ani)[0])
                n_t = len(tr_idx)
                freq[c,a,label] = n_t/cond_trials[c,a]
    
    # fig, ax = plt.subplots(1,1,figsize=(2,2))
    for c,cond in enumerate(conds):
        f = freq[c,:,:]
        ax.errorbar(np.arange(n_k) + 1 +c*0.2,np.nanmean(f,axis=0),yerr=np.nanstd(f,axis=0)/np.sqrt(len(anis)),label=cond,color=color_scheme[c],
                   linestyle='',marker='o',markersize=3)

    ax.set_ylabel('#/trial')
    ax.set_xlabel('cluster')
    ax.set_xlim(0,n_k+1)
    # ax.set_xticks(np.arange(0,n_k+2,2))
    ax.set_ylim(y_min,y_max)
    ax.set_yticks(np.arange(0,y_max+1,1))
    # ax.legend(loc=2)
    ax = xy_axis(ax)

    # if len(anis)==1:
    #     fig.suptitle(anis[0])
    # fig.tight_layout()
    
    # if save_pdf:
    #     pp.savefig(fig)
    #     plt.close(fig)

    for n in range(n_k):
        s,p = stats.wilcoxon(freq[0,:,n],freq[1,:,n])
        print('cluster %d p=%0.3f' % ((n+1),p))
    
    return ax, freq#fig, ax

def plot_movement_clusters(df,labels,side,cluster_key,condition,plot_params,save_pdf,pp):
    
    ### Plot all traces and average trace for each cluster with color matching UMAP cluster plot
    
    ### INPUTS
    ### df: the dataframe with all the data
    ### labels: the dataframe with the UMAP labels from create_umap_labels
    ### condition: the key from df that you want to split the data into (e.g. 'ocular')
    ### plot_params: a dictionary of parameters including a colormap 'cm' and colors 'cond_cols'
    ### save_pdf: whether option was selected to save pdf
    ### pp: the pdf object (only necessary if saving PDF)
    
    ### OUTPUTS
    ### fig, ax: the figure and axes objects
    
    n_k = len(np.unique(labels[cluster_key]))
    conds = np.unique(df[condition])
    cols, col_map = make_colormap(n_k)
    anis = np.unique(df['subject'])

    if side=='Side':
        side = 'Side_'
    elif side=='Top':
        side = 'Top_'

#     columns = int(np.ceil(n_k/2))
#     rows = len(conds)*2
    columns = n_k
    rows = len(conds)

    fig = plt.figure(figsize=(8,rows))
    spec = gridspec.GridSpec(ncols=columns, nrows=rows, figure=fig)
    cnt=1
    for c,cond in enumerate(conds):
        for label in range(n_k):
            tr_idx = np.intersect1d(np.where(labels[cluster_key]==label)[0],np.where(labels[condition]==cond)[0])
            sub_idx = np.sort((np.random.random(100)*len(tr_idx)).astype('int'))
            ax = fig.add_subplot(rows,columns,cnt)
#           
            for idx in tr_idx[sub_idx]:
                # rn = random.uniform(0,1)
                # if rn<=0.1:
                xtr = labels[side + 'x_eye_mvmnt'].copy()[idx] - np.mean(labels[side + 'x_eye_mvmnt'].copy()[idx])
                ytr = labels[side + 'y_eye_mvmnt'].copy()[idx] - np.mean(labels[side + 'y_eye_mvmnt'].copy()[idx])
                ax.plot(xtr,ytr,color=[0.5,0.5,0.5],linewidth=0.25,alpha=0.5)

            xtr = np.mean(labels[side + 'x_eye_mvmnt'].copy()[tr_idx],axis=0)
            xtr -= np.mean(xtr)
            ytr = np.mean(labels[side + 'y_eye_mvmnt'].copy()[tr_idx],axis=0)
            ytr -= np.mean(ytr)
            col = cols[label]
            ax.plot(xtr,ytr,color=col,linewidth=1,label=str(label))
            try:
                ax.plot(xtr[0],ytr[0],'bo',markersize=2)
                ax.plot(xtr[-1],ytr[-1],'ro',markersize=2)
            except:
                pass

            ax.set_xlim(-3,3)
            ax.set_ylim(-3,3)
            # if c==0:
            #     ax.set_title('cluster %d' % (label+1))#,oc[cond]))
#             ax.text(-2.5,-2.5,'n=%d in %d trials' % (n_t,cond_trials[c]), fontsize=10)
            ax.set_xticks(np.arange(-2,3,step=2))
            ax.set_yticks(np.arange(-2,3,step=2))
            ax.axis('off')

            cnt+=1
            
    ax = add_scalebar(ax,1,1,'cm','cm')
    if len(anis)==1:
        fig.suptitle(anis[0])
    fig.tight_layout()
    
    if save_pdf:
        pp.savefig(fig)
        plt.close(fig)
        
    return fig, ax



def filter_raw_decision(pts,like_thresh,key,row):
    x=pts[key + ' x'][int(row['Side_Start']):int(row['Side_Jump'])]
    y=pts[key + ' y'][int(row['Side_Start']):int(row['Side_Jump'])]
    like=pts[key + ' likelihood'][int(row['Side_Start']):int(row['Side_Jump'])]
    y[like<like_thresh] = np.nan
    x[like<like_thresh] = np.nan

    box_sz = 5 #for box filtering
    box = np.ones(box_sz)/box_sz

    x = pd.Series(x).interpolate().to_numpy() #interp over nans
    x = pd.Series(x).fillna(method='bfill').to_numpy() #if first vals are nans fill them
    x = signal.medfilt(x,kernel_size=3) #mean filter
    x = np.convolve(x, box, mode='same') #smoothing filter
    x = x[2:-2]

    y = pd.Series(y).interpolate().to_numpy() #interp over nans
    y = pd.Series(y).fillna(method='bfill').to_numpy() #if first vals are nans fill them
    y = signal.medfilt(y,kernel_size=3) #mean filter
    y = np.convolve(y, box, mode='same') #smoothing filter
    y = y[2:-2]
    
    return x,y


def plot_example_trial_movements(row,labels,key,cluster_key,pwin,plim,like_thresh,vid_dir,side,save_pdf,pp):
    vidname = vidname_from_row(vid_dir,side,row)
    frame,fps,frame_width,frame_height,ret = grab_vidframe(vidname,int(row['Side_Jump']))
    dlcname = dlc_file_from_row(vid_dir,side,row)
    pts = pd.read_hdf(dlcname)
    pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]

    n_mov = len(row['Side_windows'])
    n_k = len(np.unique(labels[cluster_key]))
    labels_plt = labels_df_from_df_row(labels,row)
    clusts = labels_plt[cluster_key].to_numpy()
    cols, col_map = make_colormap(n_k) #make this for clusters, load in labels
    ran = int(row['fps']*pwin*2)

    fig, axs = plt.subplots(1,2,figsize=(7,2))
    axs = axs.ravel()

    ax = axs[0]
    ax.imshow(frame)
    x,y = filter_raw_decision(pts,like_thresh,key,row)
    ax.plot(x,y,'k','-',linewidth=0.75)
    if n_mov>0:
        for w,win in enumerate(row['Side_windows']):
            x_mov = x[win:win+ran]
            y_mov = y[win:win+ran]
            ax.plot(x_mov,y_mov,'-',color=cols[clusts[w]],linewidth=1)


    ax.plot([1050,1050+row['Side_pixpercm']],[450,450],'k',linewidth=2)
    ax.axis([815,1140,750,425])
    ax.axis('off')

    # fig.tight_layout()
    # if save_pdf:
    #     pp.savefig(fig,dpi=300)
    #     plt.close(fig)


    # fig, ax = plt.subplots(1,1,figsize=(6,2))
    # axs = axs.ravel()

    x = x/row['Side_pixpercm']
    y = y/row['Side_pixpercm']

    ax = axs[1]
    ax.plot(np.arange(len(x)),x-np.mean(x)+2,'k','-',linewidth=0.75)

    # if n_mov>0:
    for w,win in enumerate(row['Side_windows']):
        x_mov = x[win:win+ran]
        ax.plot(np.arange(win,win+ran),x_mov-np.mean(x)+2,'-',color=cols[clusts[w]],linewidth=1)
        ax.plot([int(win+pwin*row['fps']),int(win+pwin*row['fps'])],[-5,5],':',color=cols[clusts[w]],linewidth=0.5,alpha=0.5)
    # ax.set_ylim(-plim,plim)
    ax.plot([len(x)-row['fps']/2,len(x)],[4,4],linewidth=2,color='k') #500ms
    ax.plot([len(x),len(x)],[4,5],linewidth=2,color='k') #1 cm
    # ax.axis('off')

    # ax = axs[1]
    ax.plot(np.arange(len(y)),-y-np.mean(-y)-2,'k','-',linewidth=0.75)
    # if n_mov>0:
    for w,win in enumerate(row['Side_windows']):
        y_mov = y[win:win+ran]
        ax.plot(np.arange(win,win+ran),-y_mov-np.mean(-y)-2,'-',color=cols[clusts[w]],linewidth=1)
    ax.set_ylim(-plim,plim)
    ax.axis('off')

    fig.tight_layout()
    if save_pdf:
        pp.savefig(fig,dpi=300)
        plt.close(fig)

    # if n_mov>0:
    fig, axs = plt.subplots(1,len(row['Side_windows']),figsize=(len(row['Side_windows']),1))
    axs = axs.ravel()
    lim=2
    for w,win in enumerate(row['Side_windows']):
        ax = axs[w]
        x_mov = x[win:win+ran]-np.mean([np.max(x[win:win+ran]),np.min(x[win:win+ran])])
        y_mov = y[win:win+ran]-np.mean([np.max(y[win:win+ran]),np.min(y[win:win+ran])])
        ax.plot(x_mov,-y_mov,'-',color=cols[clusts[w]],linewidth=0.5)
        ax.plot(x_mov[0],-y_mov[0],'bo',markersize=1)
        ax.plot(x_mov[-1],-y_mov[-1],'ro',markersize=1)
        ax.axis([-lim,lim,-lim,lim])
        ax.axis('off')
    ax.plot([0,1],[-1,-1],'k',linewidth=2) #1cm
    ax.plot([1,1],[0,-1],'k',linewidth=2) #1cm

    fig.tight_layout()
    if save_pdf:
        pp.savefig(fig)
        plt.close(fig)
    

def plot_movement_trace_distance(labels,side,cluster_num,cluster_key,condition,plot_params,save_pdf,pp):
    
    ### Plot average trace for each cluster across distances between two conditions (e.g. ocular)
    
    ### INPUTS
    ### labels: the dataframe with the UMAP labels from create_umap_labels
    ### k_clust: the k-means cluster number you want to analyze
    ### condition: the key from df that you want to split the data into (e.g. 'ocular')
    ### plot_params: a dictionary of parameters including a colormap 'cm' and colors 'cond_cols'
    ### save_pdf: whether option was selected to save pdf
    ### pp: the pdf object (only necessary if saving PDF)
    
    ### OUTPUTS
    ### fig, ax: the figure and axes objects
    
    anis = np.unique(labels['subject'])
    conds = np.unique(labels[condition])
    dists = np.unique(labels['distance_DLC'])

    if side=='Side':
        side = 'Side_'
    elif side=='Top':
        side = 'Top_'

    fig, axs = plt.subplots(1,len(dists),figsize=(2*len(dists),2))
    axs = axs.ravel()
    
    for i,d in enumerate(dists):
        ax = axs[i]
        
        for c,cond in enumerate(conds):
            tr_idx = np.intersect1d(np.where(labels[cluster_key]==cluster_num)[0],
                                    np.intersect1d(np.where(labels[condition]==cond)[0],
                                        np.intersect1d(np.where(labels['success']==1)[0],
                                            np.where(labels['distance_DLC']==d)[0])))

            xtr = labels[side + 'x_eye_mvmnt'].copy()[tr_idx]
            ytr = labels[side + 'y_eye_mvmnt'].copy()[tr_idx]

            try:
                xvals = np.mean(xtr,axis=0)
                xvals -= np.mean(xvals)
                yvals = np.mean(ytr,axis=0)
                yvals -= np.mean(yvals)
                ax.plot(xvals,yvals,color=plot_params['cond_col'][c],linewidth=3,label=cond)
                ax.plot(xvals[0],yvals[0],'bo',markersize=5)
                ax.plot(xvals[-1],yvals[-1],'ro',markersize=5)  

            except:
                pass

            ax.set_title('clust %d dist %d' % (cluster_num,d),fontsize=12)
            ax.set_xlim(-2,2)
            ax.set_ylim(-2,2)
            ax.axis('off')
    
    ax.legend(fontsize=7,loc=2)
    ax = add_scalebar(ax,1,1,'cm','cm')
    if len(anis)==1:
        fig.suptitle(anis[0])
    if save_pdf:
        pp.savefig(fig)
        plt.close(fig)

    return fig, axs


def calculate_movement_frequency(df,labels,cluster_num,cluster_key,condition,plot_params,save_pdf,pp):
    ### Calculate and plot the frequency of movement cluster 'cluster_num' at each distance
    
    ### INPUTS
    ### df: the dataframe with all of the data
    ### labels: the dataframe with the UMAP labels from create_umap_labels
    ### k_clust: the k-means cluster number you want to analyze
    ### condition: the key from df that you want to split the data into (e.g. 'ocular')
    ### plot_params: a dictionary of parameters including a colormap 'cm' and colors 'cond_cols'
    ### save_pdf: whether option was selected to save pdf
    ### pp: the pdf object (only necessary if saving PDF)
    
    ### OUTPUTS
    ### fig, ax: the figure and axes objects

    anis = np.unique(labels['subject'])
    dists = np.unique(labels['distance_DLC'])
    conds = np.unique(labels[condition])
    
    #fist get the number of trials at each distance for each animal
    dcount = np.zeros((len(anis),len(dists),len(conds)))
    for a,ani in enumerate(anis):
        for d,dist in enumerate(dists):
            for c,cond in enumerate(conds):
                dcount[a,d,c] = df[(df['subject']==ani)&(df['distance_DLC']==dist)&(df[condition]==cond)].shape[0]

    #then get the frequency of each movement type at each distance
    mcount = np.zeros((len(anis),len(dists),len(conds)))
    for a,ani in enumerate(anis):
        for d,dist in enumerate(dists):
            for c,cond in enumerate(conds):
                tot = len(np.intersect1d(np.where(labels['subject']==ani)[0],
                                              np.intersect1d(np.where(labels[cluster_key]==cluster_num)[0],
                                                             np.intersect1d(np.where(labels[condition]==cond)[0],
                                                                            np.intersect1d(np.where(labels['distance_DLC']==dist)[0],
                                                                                           np.where(labels['success']==1)[0])))))
                mcount[a,d,c] = tot/dcount[a,d,c]

    #then plot the results
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    for c,cond in enumerate(conds):
        curve = mcount[:,:,c]
        ax.errorbar(dists,np.nanmean(curve,axis=0),yerr=np.nanstd(curve,axis=0)/np.sqrt(len(anis)),
                    color=plot_params['cond_col'][c],marker='o',markersize=12,ls='',label=cond)
        
        
        x_new,y_new = plot_regression(dists,np.nanmean(curve,axis=0))
        ax.plot(x_new,y_new,':',color=plot_params['cond_col'][c],linewidth=2)
        
        
        ax.set_xlim(dists[0]-1,dists[-1]+1)
#         ax.set_ylim(0,3)
        ax.set_ylabel('mvmnt%d incidence per trial' % cluster_num)
        ax.set_xlabel('distance_DLC')
#         ax.legend(fontsize=8)
    ax.legend()
    ax = xy_axis(ax)
    if len(anis)==1:
        fig.suptitle(anis[0])
    # fig.tight_layout()

    if save_pdf:
        pp.savefig(fig)
        plt.close(fig)
        
    return fig, ax


def plot_movement_timing(labels,side,cluster_key,condition,vmax,bin_min,bin_step,pwin,plot_params,save_pdf,pp):
    ### Plot the timing movement cluster relative to the jump time
    
    ### INPUTS
    ### labels: the dataframe with the UMAP labels from create_umap_labels
    ### condition: the key from df that you want to split the data into (e.g. 'ocular')
    ### plot_params: a dictionary of parameters including a colormap 'cm' and colors 'cond_cols'
    ### save_pdf: whether option was selected to save pdf
    ### pp: the pdf object (only necessary if saving PDF)
    
    ### OUTPUTS
    ### fig, ax: the figure and axes objects
    
    labels = remove_aborts(labels)
    anis = np.unique(labels['subject'])
    dists = np.unique(labels['distance_DLC'])
    conds = np.unique(labels[condition])
    success = np.unique(labels['success'])
    if len(success)==2:
        suc_lab = ['fail','success']
    else:
        suc_lab = ['fail','success','abort']
    n_k = len(np.unique(labels[cluster_key]))
    # mov_dur = mov_win/2 #movement duration in sec
    #bins are hard coded right now
    # tjump_histrange = [tstart,0]
    # tstart_histrange = [0,1]
    bins = np.arange(bin_min,bin_step,bin_step)

    #first calculate the times for each animal/condition/distance
    tjump = np.zeros((len(anis),len(conds),n_k,len(success),int(np.abs(bin_min/bin_step))))
    # tstart = np.zeros((len(anis),len(conds),n_k,bins))
    for a,ani in enumerate(anis):
        for c,cond in enumerate(conds):
            for s,suc in enumerate(success):
                tot_tr = len(labels[(labels['subject']==ani)&(labels[condition]==cond)&(labels['success']==suc)])
                for k in range(n_k):
                    tr_idx = np.intersect1d(np.where(labels['subject']==ani)[0],
                                                  np.intersect1d(np.where(labels[cluster_key]==k)[0],
                                                                 np.intersect1d(np.where(labels[condition]==cond)[0],
                                                                                np.where(labels['success']==suc)[0])))
                    tjump[a,c,k,s,:] = np.histogram(labels[side + '_mov_rel_jump'][tr_idx]+(pwin*2),
                                                  bins=bins)[0]/tot_tr#len(labels[side + '_mov_rel_jump'][tr_idx])
        #             tstart[a,c,k,:] = np.histogram(labels[side + '_mov_rel_start'][tr_idx],
        #                                            bins=bins,range=tstart_histrange)[0]/len(labels[side + '_mov_rel_start'][tr_idx])

    #then plot the results for time from jump
    fig, axs = plt.subplots(len(success),len(conds),figsize=(2*len(conds),2*len(success)))
    axs = axs.ravel()

    for c,cond in enumerate(conds):
        for s,suc in enumerate(success):
            ax = axs[c+len(conds)*s]
            hist = np.nanmean(tjump[:,c,:,s,:],axis=0)
            pos = ax.imshow(hist,cmap=plot_params['cm'],vmin=0,vmax=vmax)
            ax.set_yticks(np.arange(0,n_k,1))
            ax.set_xticks(np.arange(0,np.abs(bin_min/bin_step),np.abs(bin_min)))
            ax.set_xticklabels(np.arange(bin_min,0,np.abs(bin_step*bin_min)))
            ax.set_title(cond + ' ' + suc_lab[s])
            cbar = fig.colorbar(pos, ax=ax, orientation='vertical')
            cbar.ax.tick_params(labelsize=10) 
            if c==0:
                ax.set_ylabel('%s num' % cluster_key)
            if s==len(success)-1:
                ax.set_xlabel('time before jump (s)')
            if c==len(conds)-1:
                cbar.ax.set_ylabel('norm. movement freq.',rotation=90,fontsize=12)

    fig.tight_layout()
    if len(anis)==1:
        fig.suptitle(anis[0])
    # fig.tight_layout()

    if save_pdf:
        pp.savefig(fig)
        plt.close(fig)
    
    ### this is commented out for now but will plot relative to start time

#     #then plot the results for time from start
#     fig2, axs = plt.subplots(1,len(conds),figsize=(10*len(conds),10))
#     axs2 = axs.ravel()
#     for c,cond in enumerate(conds):
#         hist = np.nanmean(tstart[:,c,:,:],axis=0)
#         ax = axs2[c]
#         pos = ax.imshow(hist,cmap=plot_params['cm'])
#         ax.set_xticks(np.arange(0,13,2))
#         ax.set_xticklabels(np.round(np.arange(0,1+2/bins,2/bins),2))
#         ax.set_title(cond)
#         ax.set_xlabel('normalized trial time (s)')
#         ax.set_ylabel('movement type')

#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes('right', size='5%', pad=0.05)
#     cbar = fig.colorbar(pos, cax=cax, orientation='vertical')
#     cbar.ax.set_ylabel('normalized movement frequency',rotation=90,fontsize=15)
#     fig2.tight_layout()

#     if save_pdf:
#         pp.savefig(fig2)
        # plt.close(fig2)

    #sum across rows to make sure this is correct (adds to 1 w/in animal)
    
    return fig, axs



### plot of movement timing normalized to the total number of trials in each condition
def plot_movement_timing_norm_tot_trials(og_df,st_remap,cond_lab,state_key_vals,state_key_times,side,bin_min,bin_step,save_pdf,pp):
    base_df = og_df.copy()
    conds = np.unique(base_df[cond_lab])
    anis = np.unique(base_df['subject'])
    dists = np.unique(base_df['distance_DLC'])

    n_k = len(st_remap) #len(np.unique(state_list))-1 #b/c -1 makes it one longer
    bins = np.arange(bin_min,bin_step,bin_step)
    
    plt_col = ['k',[0.5,0.5,0.5]]
    suc_lab = ['fail','success','abort']
    fig, axs = plt.subplots(3,3,figsize=(20,int(len(st_remap)*2)))

    st_hist = np.empty((n_k,int(np.abs(bin_min/bin_step)),len(anis),len(conds),len(np.unique(base_df['success']))))
    st_hist[:] = np.nan

    for suc in np.unique(base_df['success']):
        for c,cond in enumerate(conds):

            for a,ani in enumerate(anis):
                df = base_df[(base_df[cond_lab]==cond)&(base_df['success']==suc)&(base_df['subject']==ani)]
                df.reset_index(inplace=True,drop=True)
                if len(df)>0:
                    state_list = np.array(flatten_list(df[state_key_vals].to_list()))
                    state_times = np.array(flatten_list(df[state_key_times].to_list()))
                    
                    trunc_list = state_list[state_times>bin_min]
                    trunc_times = state_times[state_times>bin_min]
                    
                    

                    # mn_t, sd_t, plt_t = ([] for i in range(3))
                    for s,state in enumerate(st_remap):
                        t = trunc_times[np.where(trunc_list==state)[0]]
                        hist = np.histogram(t,bins=bins)[0]
                        st_hist[s,:,a,c,suc] = hist/df.shape[0]
                    #     try:
                    #         mn_t.append(np.mean(t))
                    #         sd_t.append(np.std(t))
                    #         plt_t.append(np.where(np.mean(t)>bins)[0][-1]) # find the bin with the mean value for plotting
                    #     except:
                    #         mn_t.append(np.nan)
                    #         sd_t.append(np.nan)
                    #         plt_t.append(np.nan)
                    # mn_t = np.array(mn_t)
                    # sd_t = np.array(sd_t)
                    # plt_t = np.array(plt_t)

            ax = axs[suc,0]
            ax.errorbar(x=np.arange(n_k),y=np.nanmean(np.nansum(st_hist[:,:,:,c,suc],axis=1),axis=1),
                yerr=np.nanstd(np.nansum(st_hist[:,:,:,c,suc],axis=1),axis=1),
                marker='o',markersize=10,color=plt_col[c],linestyle='',label=cond)
            ax.set_ylabel('#/trial')
            ax.set_xticks(np.arange(0,len(st_remap)))
            ax.set_xticklabels(np.arange(1,len(st_remap)+1))
            ax.set_xlabel('state')
            ax.set_ylim(0,5)
            ax = xy_axis(ax)

            ax = axs[suc,c+1]
            p0 = ax.imshow(np.nanmean(st_hist[:,:,:,c,suc],axis=2),cmap='jet',vmin=0,vmax=0.5)#[np.argsort(mn_t),:]
            # ax.plot(plt_t,np.arange(n_k),'w*',markersize=10,linewidth=0)
            ax.set_xticks(np.arange(0,np.abs(bin_min/bin_step),np.abs(bin_min)))
            ax.set_xticklabels(np.arange(bin_min,0,np.abs(bin_step*bin_min)))
            ax.set_yticks(np.arange(0,len(st_remap)))
            ax.set_yticklabels(np.arange(1,len(st_remap)+1))
            ax.set_ylabel('state #')
            ax.set_xlabel('time before jump (s)')
            # ax.set_yticks(np.arange(0,n_k))
        #     ax.set_yticklabels(np.argsort(mn_t))
            ax.set_title(cond + ' ' + suc_lab[suc])
            fig.colorbar(p0,ax=ax,label='relative frequency')

        # for st in range(n_k):
        #     for d,dist in enumerate(dists):
        #         s, p = stats.ttest_ind(stats_array[st,d,0],stats_array[st,d,1])
        #         print('state %d, distance %d, p=%0.3f' % ((st+1),dist,p))
        # print('alpha=%0.3f' % (0.05/len(dists)))

    axs[suc,0].legend(fontsize=8) 
    fig.tight_layout()

    for s in range(len(st_remap)):
        binoc = np.sum(st_hist[s,:,:,0,1],axis=0)
        monoc = np.sum(st_hist[s,:,:,1,1],axis=0)
        stat, pval = stats.ttest_rel(binoc,monoc)
        print('state %d binoc vs. monoc freq p=%0.3f' % (s+1,pval))
    print('alpha = %0.3f' % (0.05/len(st_remap)))
    
    if save_pdf:
        pp.savefig(fig)
        plt.close(fig)
        
    return fig, axs, st_hist#mn_t, sd_t


save_pdf = False

def plot_cluster_amplitudes(labels,cluster_key,condition,ymin,ymax,color_scheme,save_pdf,pp):
    ### Calculate and plot the amplitude of movement cluster 'cluster_num' at each distance
    
    ### INPUTS
    ### labels: the dataframe with the UMAP labels from create_umap_labels
    ### condition: the key from df that you want to split the data into (e.g. 'ocular')
    ### k_clust: the k-means cluster number you want to analyze
    ### success: 0 or 1 for if you want success or failure trials
    ### plot_params: a dictionary of parameters including a colormap 'cm' and colors 'cond_cols'
    ### save_pdf: whether option was selected to save pdf
    ### pp: the pdf object (only necessary if saving PDF)
    
    ### OUTPUTS
    ### fig, ax: the figure and axes objects
    anis = np.unique(labels['subject'])
    conds = np.unique(labels[condition])

    labels = labels[labels['success']==1]

    n_k = len(np.unique(labels[cluster_key]))

    #first calculate the amplitudes
    amps = np.zeros((len(anis),len(conds),n_k))
    for a,ani in enumerate(anis):
        for c,cond in enumerate(conds):
            for cluster_num in range(n_k):
                tr_idx = np.intersect1d(np.where(labels['subject']==ani)[0],np.intersect1d(np.where(labels[cluster_key]==cluster_num)[0],np.where(labels[condition]==cond)[0]))

                amps[a,c,cluster_num] = np.nanmean(labels['xamps_mm'].to_numpy()[tr_idx])
                amps[a,c,cluster_num] = np.nanmean(labels['yamps_mm'].to_numpy()[tr_idx])
                
    #then plot the results
    fig, axs = plt.subplots(1,1,figsize=(2,2))
    for c,cond in enumerate(conds):
            axs.errorbar(x=np.arange(n_k)+1,y=np.nanmean(amps[:,c,:],axis=0),yerr=np.nanstd(amps[:,c,:],axis=0)/np.sqrt(len(anis)),color=color_scheme[c],linestyle='',marker='o',markersize=3)

    axs.axis([0,n_k+1,ymin,ymax])
    axs.set_xlabel('cluster')
    axs.set_ylabel('amplitude (cm)')
    axs.set_xticks(np.arange(0,n_k+n_k/2,n_k/2))
    # axs.set_yticks([0,0.5,1.5])

    axs = xy_axis(axs)
    fig.tight_layout()

    if save_pdf:
        pp.savefig(fig)
        plt.close(fig)

    for cluster_num in range(n_k):
        s, p = stats.ttest_rel(amps[:,0,cluster_num],amps[:,1,cluster_num])
        print('binoc vs. monoc amp cluster %d p=%0.2f' % (cluster_num+1,p))


    return fig, axs, amps



def plot_movement_amplitudes(labels,cluster_num,cluster_key,condition,plot_params,save_pdf,pp):
    ### Calculate and plot the amplitude of movement cluster 'cluster_num' at each distance
    
    ### INPUTS
    ### labels: the dataframe with the UMAP labels from create_umap_labels
    ### condition: the key from df that you want to split the data into (e.g. 'ocular')
    ### k_clust: the k-means cluster number you want to analyze
    ### success: 0 or 1 for if you want success or failure trials
    ### plot_params: a dictionary of parameters including a colormap 'cm' and colors 'cond_cols'
    ### save_pdf: whether option was selected to save pdf
    ### pp: the pdf object (only necessary if saving PDF)
    
    ### OUTPUTS
    ### fig, ax: the figure and axes objects
    
    labels = remove_aborts(labels)
    
    anis = np.unique(labels['subject'])
    dists = np.unique(labels['distance_DLC'])
    conds = np.unique(labels[condition])
    success = np.unique(labels['success'])

    #first calculate the amplitudes
    amps = np.zeros((2,len(anis),len(dists),len(conds),len(success)))
    for a,ani in enumerate(anis):
        for d,dist in enumerate(dists):
            for c,cond in enumerate(conds):
                for s,suc in enumerate(success):
                    tr_idx = np.intersect1d(np.where(labels['subject']==ani)[0],
                                                  np.intersect1d(np.where(labels[cluster_key]==cluster_num)[0],
                                                                 np.intersect1d(np.where(labels[condition]==cond)[0],
                                                                                np.intersect1d(np.where(labels['distance_DLC']==dist)[0],
                                                                                               np.where(labels['success']==suc)[0]))))

                    amps[0,a,d,c,s] = np.nanmean(labels['xamps'][tr_idx])
                    amps[1,a,d,c,s] = np.nanmean(labels['yamps'][tr_idx])
                
    #then plot the results
    fig, axs = plt.subplots(1,3,figsize=(15,5))
    pltlabel = ['x-amp','y-amp','y/x ratio']
    leglabel = ['fail','success']
    cnt=0
    for c,cond in enumerate(conds):
        for s,suc in enumerate(success):
            for sub in range(3):
                if sub==2:
                    curve = amps[1,:,:,c,s]/amps[0,:,:,c,s]
                else:
                    curve = amps[sub,:,:,c,s]
                ax = axs[sub]
                ax = xy_axis(ax)
                try:
                    pr = stats.pearsonr(dists,np.nanmean(curve,axis=0))[0]
                    x_new,y_new = plot_regression(dists,np.nanmean(curve,axis=0))
                    ax.plot(x_new,y_new,':',color=plot_params['4cond_col'][c],linewidth=2)
                except:
                    pr = 0.000
                ax.errorbar(dists,np.nanmean(curve,axis=0),yerr=np.nanstd(curve,axis=0)/np.sqrt(len(anis)),
                            color=plot_params['4cond_col'][cnt],marker='o',markersize=12,ls='',
                            label='%s success=%s r2=%0.2f' % (cond,suc,pr))

                if c==len(conds)-1:
                    ax.set_xlim(dists[0]-1,dists[-1]+1)
                    ax.set_ylim(0,3)
                    ax.set_ylabel('mvmnt%d %s' % (cluster_num,pltlabel[sub]))
                    ax.set_xlabel('distance_DLC')
                    ax.legend(fontsize=8)
            cnt+=1

    if len(anis)==1:
        fig.suptitle(anis[0])
    # fig.tight_layout()

    if save_pdf:
        pp.savefig(fig)
        plt.close(fig)
    
    return fig, axs



def plot_yamps(labels,cluster_num,cluster_key,condition,success,ylim,plot_params,save_pdf,pp):
    ### Calculate and plot the amplitude of movement cluster 'cluster_num' at each distance for y amplitude only
    
    ### INPUTS
    ### labels: the dataframe with the UMAP labels from create_umap_labels
    ### condition: the key from df that you want to split the data into (e.g. 'ocular')
    ### k_clust: the k-means cluster number you want to analyze
    ### success: 0 or 1 for if you want success or failure trials
    ### plot_params: a dictionary of parameters including a colormap 'cm' and colors 'cond_cols'
    ### save_pdf: whether option was selected to save pdf
    ### pp: the pdf object (only necessary if saving PDF)
    
    ### OUTPUTS
    ### fig, ax: the figure and axes objects
    
    anis = np.unique(labels['subject'])
    dists = np.unique(labels['distance_DLC'])
    conds = np.unique(labels[condition])

    #first calculate the amplitudes
    amps = np.zeros((len(anis),len(dists),len(conds)))
    for a,ani in enumerate(anis):
        for d,dist in enumerate(dists):
            for c,cond in enumerate(conds):
                tr_idx = np.intersect1d(np.where(labels['subject']==ani)[0],
                                              np.intersect1d(np.where(labels[cluster_key]==cluster_num)[0],
                                                             np.intersect1d(np.where(labels[condition]==cond)[0],
                                                                            np.intersect1d(np.where(labels['distance_DLC']==dist)[0],
                                                                                           np.where(labels['success']==success)[0]))))

                amps[a,d,c] = np.nanmean(labels['yamps'][tr_idx])
                
    #then plot the results
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    pltlabel = ['x-amp','y-amp','y/x ratio']
    for c,cond in enumerate(conds):
        curve = amps[:,:,c]
        try:
            pr = stats.pearsonr(dists,np.nanmean(curve,axis=0))[0]
            x_new,y_new = plot_regression(dists,np.nanmean(curve,axis=0))
            ax.plot(x_new,y_new,':',color=plot_params['cond_col'][c],linewidth=2)
        except:
            pr = 0.000
        ax.errorbar(dists,np.nanmean(curve,axis=0),yerr=np.nanstd(curve,axis=0)/np.sqrt(len(anis)),
                    color=plot_params['cond_col'][c],marker='o',markersize=12,ls='',label='%s r2=%0.2f' % (cond,pr))
        if c==len(conds)-1:
            ax.set_xlim(dists[0]-1,dists[-1]+1)
            ax.set_ylim(0,ylim)
            ax.set_yticks(np.arange(0,ylim+1,1))
            ax.set_ylabel('mvmnt%d y-amp' % (cluster_num))
            ax.set_xlabel('distance_DLC')
            ax.legend()

    ax = xy_axis(ax)
    if len(anis)==1:
        fig.suptitle(anis[0])
    # fig.tight_layout()

    if save_pdf:
        pp.savefig(fig)
        plt.close(fig)
    
    return fig, ax



def get_movement_amplitudes_mm(labels,side):

    ### Adds columns to labels with the x and y amplitudes for each individual movement

    ### INPUTS
    ### labels: the dataframe created by create_umap_labels

    ### OUTPUTS
    ### labels: the same dataframe with two new columns appended, xamps and yamps

    xamps = []
    yamps = []
    visangle = []
    for index,row in labels.iterrows():

        xtr = row[side + '_x_eye_mvmnt'].copy()
        ytr = row[side + '_y_eye_mvmnt'].copy()

        xamps.append(np.max(xtr)-np.min(xtr))
        yamps.append(np.max(ytr)-np.min(ytr))

        dist = row['distance_DLC']
        visangle.append(np.round(np.degrees(np.arctan(yamps[-1]/dist)),1))


    labels['xamps_mm'] = xamps
    labels['yamps_mm'] = yamps
    labels['visangle_mm'] = visangle

    return labels


def plot_yamps_mm(labels,cluster_num,cluster_key,condition,success,ylim,plot_params,save_pdf,pp):
    ### Calculate and plot the amplitude of movement cluster 'cluster_num' at each distance for y amplitude only
    
    ### INPUTS
    ### labels: the dataframe with the UMAP labels from create_umap_labels
    ### condition: the key from df that you want to split the data into (e.g. 'ocular')
    ### k_clust: the k-means cluster number you want to analyze
    ### success: 0 or 1 for if you want success or failure trials
    ### plot_params: a dictionary of parameters including a colormap 'cm' and colors 'cond_cols'
    ### save_pdf: whether option was selected to save pdf
    ### pp: the pdf object (only necessary if saving PDF)
    
    ### OUTPUTS
    ### fig, ax: the figure and axes objects
    
    anis = np.unique(labels['subject'])
    dists = np.unique(labels['distance_DLC'])
    conds = np.unique(labels[condition])

    #first calculate the amplitudes
    amps = np.zeros((len(anis),len(dists),len(conds)))
    for a,ani in enumerate(anis):
        for d,dist in enumerate(dists):
            for c,cond in enumerate(conds):
#                 tr_idx = np.intersect1d(np.where(labels['subject']==ani)[0],
#                                               np.intersect1d(np.where(labels[cluster_key]==cluster_num)[0],
#                                                              np.intersect1d(np.where(labels[condition]==cond)[0],
#                                                                             np.intersect1d(np.where(labels['distance_DLC']==dist)[0],
#                                                                                            np.where(labels['success']==success)[0]))))
                tr_idx = np.intersect1d(np.where(labels['subject']==ani)[0],
                                              np.intersect1d(np.where(labels[cluster_key]==cluster_num)[0],
                                                             np.intersect1d(np.where(labels[condition]==cond)[0],
                                                                           np.where(labels['distance_DLC']==dist)[0])))

                amps[a,d,c] = np.nanmean(labels['yamps_mm'][tr_idx])
                
    #then plot the results
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    pltlabel = ['x-amp','y-amp','y/x ratio']
    for c,cond in enumerate(conds):
        curve = amps[:,:,c]
        try:
            pr = stats.pearsonr(dists,np.nanmean(curve,axis=0))[0]
            x_new,y_new = plot_regression(dists,np.nanmean(curve,axis=0))
            ax.plot(x_new,y_new,':',color=plot_params['cond_col'][c],linewidth=2)
        except:
            pr = 0.000
        ax.errorbar(dists,np.nanmean(curve,axis=0),yerr=np.nanstd(curve,axis=0)/np.sqrt(len(anis)),
                    color=plot_params['cond_col'][c],marker='o',markersize=12,ls='',label='%s r2=%0.2f' % (cond,pr))
        
        if c==len(conds)-1:
            ax.set_xlim(dists[0]-1,dists[-1]+1)
            ax.set_ylim(0,ylim)
            ax.set_yticks(np.arange(0,ylim+1,1))
            ax.set_ylabel('mvmnt%d max-min y-amp' % (cluster_num))
            ax.set_xlabel('distance_DLC')
            ax.legend()

    ax = xy_axis(ax)
    if len(anis)==1:
        fig.suptitle(anis[0])
    # fig.tight_layout()

    if save_pdf:
        pp.savefig(fig)
        plt.close(fig)
    
    return fig, ax



def plot_visangles(labels,cluster_num,cluster_key,condition,success,plot_params,save_pdf,pp):
    ### Calculate and plot the frequency of movement cluster k_clust at each distance
    
    ### INPUTS
    ### labels: the dataframe with the UMAP labels from create_umap_labels
    ### condition: the key from df that you want to split the data into (e.g. 'ocular')
    ### k_clust: the k-means cluster number you want to analyze
    ### success: 0 or 1 for if you want success or failure trials
    ### plot_params: a dictionary of parameters including a colormap 'cm' and colors 'cond_cols'
    ### save_pdf: whether option was selected to save pdf
    ### pp: the pdf object (only necessary if saving PDF)
    
    ### OUTPUTS
    ### fig, ax: the figure and axes objects
    
    anis = np.unique(labels['subject'])
    dists = np.unique(labels['distance_DLC'])
    conds = np.unique(labels[condition])

    #first calculate the angles
    angles = np.zeros((len(anis),len(dists),len(conds)))
    for a,ani in enumerate(anis):
        for d,dist in enumerate(dists):
            for c,cond in enumerate(conds):
                tr_idx = np.intersect1d(np.where(labels['subject']==ani)[0],
                                              np.intersect1d(np.where(labels[cluster_key]==cluster_num)[0],
                                                             np.intersect1d(np.where(labels[condition]==cond)[0],
                                                                            np.intersect1d(np.where(labels['distance_DLC']==dist)[0],
                                                                                           np.where(labels['success']==success)[0]))))

                angles[a,d,c] = np.nanmean(labels['visangle'][tr_idx])
                
    #then plot the results
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    for c,cond in enumerate(conds):
        curve = angles[:,:,c]
        try:
            pr = stats.pearsonr(dists,np.nanmean(curve,axis=0))[0]
            x_new,y_new = plot_regression(dists,np.nanmean(curve,axis=0))
            ax.plot(x_new,y_new,':',color=plot_params['cond_col'][c],linewidth=2)
        except:
            pr = 0.000
        ax.errorbar(dists,np.nanmean(curve,axis=0),yerr=np.nanstd(curve,axis=0)/np.sqrt(len(anis)),
                    color=plot_params['cond_col'][c],marker='o',markersize=12,ls='',
                    label='%s r2=%0.2f' % (cond,pr))

        if c==len(conds)-1:
            ax.set_xlim(dists[0]-1,dists[-1]+1)
            ax.set_ylim(0,10)
            ax.set_ylabel('mvmnt%d visangle (deg)' % (cluster_num))
            ax.set_xlabel('distance_DLC')
            ax.legend()

    ax = xy_axis(ax)
    if len(anis)==1:
        fig.suptitle(anis[0])
    # fig.tight_layout()

    if save_pdf:
        pp.savefig(fig)
        plt.close(fig)
    
    return fig, ax


def plot_performance_vs_movement(df_in,labels,cluster_num,cluster_key,condition,plot_params,save_pdf,pp):
    ### Calculate and plot the frequency of movement cluster k_clust at each distance
    
    ### INPUTS
    ### df: the dataframe with all of the data
    ### labels: the dataframe with the UMAP labels from create_umap_labels
    ### k_clust: the k-means cluster number you want to analyze
    ### condition: the key from df that you want to split the data into (e.g. 'ocular')
    ### plot_params: a dictionary of parameters including a colormap 'cm' and colors 'cond_cols'
    ### save_pdf: whether option was selected to save pdf
    ### pp: the pdf object (only necessary if saving PDF)
    
    ### OUTPUTS
    ### fig, ax: the figure and axes objects
    
    anis = np.unique(labels['subject'])
    dists = np.unique(labels['distance_DLC'])
    conds = np.unique(labels[condition])

    fig, axs = plt.subplots(2,len(conds),figsize=(len(conds)*5,10))
    axs = axs.ravel()
    boblab = ['no mvmnt','mvmnt']
    df = remove_aborts(df_in)
    for c,cond in enumerate(conds):
        ax = axs[c]
        for bob in np.unique(df['bob_trial']):
            jumpcurve = df[(df['bob_trial']==bob)&(df[condition]==cond)].groupby(['subject','distance_DLC']).mean()
            jumpcurve.reset_index(inplace=True)
            mnplot = jumpcurve.groupby(['distance_DLC']).mean()
            semplot = jumpcurve.groupby(['distance_DLC']).std()/np.sqrt(len(anis))
            mnplot.reset_index(inplace=True)
            semplot.reset_index(inplace=True)

            ax.errorbar(mnplot['distance_DLC'],
                     mnplot['success'],
                     yerr=semplot['success'],label=boblab[bob],color=plot_params['cond_col'][int(bob)])

        ax.set_xlabel('gap distance (cm)')
        ax.set_ylabel('success rate')
        ax.set_title('%s mvmnt%s' % (cond,cluster_num))
        ax.set_xlim(6,29)
        ax.set_ylim(0,1.1)
        ax.legend(fontsize=10)
        plt.xticks(np.arange(10, 30, step=5))
        plt.yticks(np.arange(0, 1.5, step=0.5))
        ax = xy_axis(ax)

    df = remove_aborts(df_in)
    for c,cond in enumerate(conds):
        ax = axs[c+2]
        for bob in np.unique(df['bob_trial']):
            jumpcurve = df[(df['bob_trial']==bob)&(df[condition]==cond)].groupby(['subject','distance_DLC','jumpdist']).mean()
            jumpcurve.reset_index(inplace=True)
            mnplot = jumpcurve.groupby(['distance_DLC']).mean()
            semplot = jumpcurve.groupby(['distance_DLC']).std()/np.sqrt(len(anis))
            mnplot.reset_index(inplace=True)
            semplot.reset_index(inplace=True)

            ax.errorbar(mnplot['distance_DLC'],
                     mnplot['jumpdist'],
                     yerr=semplot['jumpdist'],label=boblab[bob],color=plot_params['cond_col'][int(bob)])

        ax.plot([5,35],[5,35],'k:')
        ax.set_xlabel('gap distance (cm)')
        ax.set_ylabel('distance jumped (cm)')
        ax.set_title('%s mvmnt%s' % (cond,cluster_num))
        ax.set_xlim(5,35)
        ax.set_ylim(5,35)
        ax.legend(fontsize=10)
        ax = xy_axis(ax)
    if len(anis)==1:
        fig.suptitle(anis[0])
    # plt.tight_layout()

    if save_pdf:
        pp.savefig(fig)
        plt.close(fig)
        
    return fig, ax


def make_ris_table(pfs,ds,save_pdf,pp):
    
    ### Create a lookup table for retinal image sizes for platform/distance combinations
    
    ### INPUTS
    ### pfs: an array containing the platform sizes in cm
    ### ds: an array containing the gap distances in cm
    ### save_pdf: whether option was selected to save pdf
    ### pp: the pdf object (only necessary if saving PDF)
    
    ### OUTPUTS
    ### ris: table of retinal image sizes
    ### fig, ax: the figure and axes objects

    #first gen platform/distance parameters
    ris = np.degrees(np.arctan(pfs/ds[:,None])).T

#     plt.rcParams.update({'font.size': 20})

    fig, ax = plt.subplots(figsize=(len(ds)*2,len(pfs)*2))
    im = ax.imshow(ris,cmap='cool')

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(ds)))
    ax.set_yticks(np.arange(len(pfs)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(ds)
    ax.set_yticklabels(pfs)

    fs = 20 #font size for plotting
    # Loop over data dimensions and create text annotations.
    for i in range(len(ds)):
        for j in range(len(pfs)):
            text = ax.text(i, j, '%.1f' % ris[j, i],
                           ha="center", va="center", color="k",fontsize=fs)

    ax.set_title("Visual Angle (degrees)",fontsize=fs)
    ax.set_ylabel('platform width (cm)',fontsize=fs)
    ax.set_xlabel('gap distance (cm)',fontsize=fs)

    # fig.tight_layout()
    # plt.show()

    if save_pdf:
        pp.savefig(fig)
        plt.close(fig)
        
    return ris, fig, ax


def plot_performance(ax,og_df,condition,aborts,plt_min,plt_max,color_scheme):#,save_pdf,pp,suptitle=''):
    
    ### Plot success versus distance for two conditions
    
    ### INPUTS
    ### df: the dataframe containing all of the data
    ### condition: the key from df that you want to split the data into (e.g. 'ocular')
    ### plot_params: a dictionary of parameters including a colormap 'cm' and colors 'cond_cols'
    ### save_pdf: whether option was selected to save pdf
    ### pp: the pdf object (only necessary if saving PDF)
    
    ### OUTPUTS
    ### fig, ax: the figure and axes objects
    if aborts:
        df = aborts_as_failures(og_df)
    else: 
        df = remove_aborts(og_df)
    # df = aborts_as_failures(og_df)
    anis = np.unique(df['subject'])
    dists = np.unique(df['distance_DLC'])
    conds = np.unique(df[condition])

    # stats_array = np.empty((len(anis),len(dists),len(conds)))
    # stats_array[:] = np.nan

    anova_df = pd.DataFrame(columns=['distance_DLC','condition','success'])

    # fig, ax = plt.subplots(1,1,figsize=(5,5))
    for c,cond in enumerate(conds):
        jumpcurve = df[df[condition]==cond].groupby(['distance_DLC','subject']).mean()
        # jumpcurve.reset_index(inplace=True)
        mnplot = jumpcurve.groupby(['distance_DLC']).mean()
        semplot = jumpcurve.groupby(['distance_DLC']).std()/np.sqrt(len(anis))
        mnplot.reset_index(inplace=True)
        semplot.reset_index(inplace=True)

        ax.errorbar(mnplot['distance_DLC'],
                 mnplot['success'],
                 yerr=semplot['success'],label=cond,color=color_scheme[c],linewidth=1)

        # jumpcurve.reset_index(inplace=True)   
        # for ani in anis:
        #     ax.plot(jumpcurve[jumpcurve['subject']==ani]['distance_DLC'],jumpcurve[jumpcurve['subject']==ani]['success'],'-',color=color_scheme[c],linewidth=0.25,alpha=0.5)
        # print(jumpcurve)
        for d,dist in enumerate(dists):
            # print(jumpcurve.xs(dist)['success'])
            # stats_array[:,d,c] = jumpcurve.xs(dist)['success']

            temp_df = pd.DataFrame(columns=['distance_DLC','condition','success'])
            temp_df['distance_DLC']  = pd.Series(np.repeat(dist,len(anis)))
            temp_df['condition'] = pd.Series(np.repeat(cond,len(anis)))
            temp_df['success'] = pd.Series(jumpcurve.xs(dist)['success'].to_numpy())
            anova_df = pd.concat([anova_df,temp_df],axis=0)

    ax.set_xlabel('gap distance (cm)')
    ax.set_ylabel('success rate')
    ax.set_xlim(plt_min,plt_max)
    ax.set_xticks([10,15,20,25])
    # ax.set_xticks(np.arange(plt_min,plt_max+plt_min,plt_min))
    ax.set_ylim(0,1.1)
    ax.set_yticks(np.arange(0, 1.5, step=0.5))
    ax.legend(fontsize=8)
    ax = xy_axis(ax)
    # if len(anis)==1:
    #     fig.suptitle(anis[0])
    # fig.suptitle(suptitle)
    # if save_pdf:
    #     pp.savefig(fig)
    #     plt.close(fig)

    # print anova results
    model = ols('success ~ C(distance_DLC) + C(condition) + C(distance_DLC):C(condition)', data=anova_df).fit()
    print(conds)
    print(sm.stats.anova_lm(model, typ=2))
    print('')

    # for d,dist in enumerate(dists):
    #     s, p = stats.ttest_rel(stats_array[:,d,0],stats_array[:,d,1])
    #     print('distance %d, p=%0.3f' % (dist,p))
    # print('alpha=%0.3f' % (0.05/len(dists)))
    return ax, anova_df#, stats_array


def plot_jumpdist(ax,og_df,condition,plt_min,plt_max,color_scheme):#,save_pdf,pp,suptitle=''):
    
    ### Plot success versus distance for two conditions
    
    ### INPUTS
    ### df: the dataframe containing all of the data
    ### condition: the key from df that you want to split the data into (e.g. 'ocular')
    ### plot_params: a dictionary of parameters including a colormap 'cm' and colors 'cond_cols'
    ### save_pdf: whether option was selected to save pdf
    ### pp: the pdf object (only necessary if saving PDF)
    
    ### OUTPUTS
    ### fig, ax: the figure and axes objects
    df = remove_aborts(og_df)
    anis = np.unique(df['subject'])
    dists = np.unique(df['distance_DLC'])
    conds = np.unique(df[condition])

    stats_array = np.empty((len(anis),len(dists),len(conds)))

    anova_df = pd.DataFrame(columns=['distance_DLC','condition','jumpdist'])

    # fig, ax = plt.subplots(1,1,figsize=(5,5))        
    for c,cond in enumerate(conds):
        jumpcurve = df[(df['success']==1)&(df[condition]==cond)].groupby(['distance_DLC','subject']).mean()
        # jumpcurve.reset_index(inplace=True)
        mnplot = jumpcurve.groupby(['distance_DLC']).mean()
        semplot = jumpcurve.groupby(['distance_DLC']).std()/np.sqrt(len(anis))
        mnplot.reset_index(inplace=True)
        semplot.reset_index(inplace=True)
        semplot[np.isnan(semplot)]=0
        xvals = np.unique(mnplot['distance_DLC'])
        ax.errorbar(xvals,
                     mnplot['jumpdist'],
                     yerr=semplot['jumpdist'],label='%s' % cond,color=color_scheme[c],linewidth=1)

        # jumpcurve.reset_index(inplace=True)   
        # for ani in anis:
        #     ax.plot(jumpcurve[jumpcurve['subject']==ani]['distance_DLC'],jumpcurve[jumpcurve['subject']==ani]['jumpdist'],'-',color=color_scheme[c],linewidth=0.25,alpha=0.5)
        
        ax.set_xlabel('gap distance (cm)')
        ax.set_ylabel('distance jumped (cm)')
        ax.set_xlim(plt_min,plt_max)
        ax.set_ylim(plt_min,plt_max)
        ax.set_xticks([10,15,20,25])
        ax.set_yticks([10,15,20,25])
        # ax.set_xticks(np.arange(plt_min,plt_max+plt_min,plt_min))
        # ax.set_yticks(np.arange(plt_min,plt_max+plt_min,plt_min))
        
        # ax1.set_title('both bi/monocular')
        # locs, labels = plt.xticks(fontsize=10)
        # plt.xticks(np.arange(8, 32, step=4))
        # for pl in range(3):
        ax.plot(xvals,xvals,'k:')
        ax = xy_axis(ax)

        for d,dist in enumerate(dists):
            stats_array[:,d,c] = jumpcurve.xs(dist)['jumpdist']

            temp_df = pd.DataFrame(columns=['distance_DLC','condition','jumpdist'])
            temp_df['distance_DLC']  = pd.Series(np.repeat(dist,len(anis)))
            temp_df['condition'] = pd.Series(np.repeat(cond,len(anis)))
            temp_df['jumpdist'] = pd.Series(jumpcurve.xs(dist)['jumpdist'].to_numpy())
            anova_df = pd.concat([anova_df,temp_df],axis=0)

    # if len(anis)==1:
    #     fig.suptitle(anis[0])
    # fig.tight_layout()
    # fig.suptitle(suptitle)
    # if save_pdf:
    #     pp.savefig(fig)
    #     plt.close(fig)

    #print anova results
    # print(anova_df)
    model = ols('jumpdist ~ C(distance_DLC) + C(condition) + C(distance_DLC):C(condition)', data=anova_df).fit()
    print(cond)
    print(sm.stats.anova_lm(model, typ=2))
    print('')

    # for d,dist in enumerate(dists):
    #     s, p = stats.ttest_rel(stats_array[:,d,0],stats_array[:,d,1])
    #     print('distance %d, p=%0.3f' % (dist,p))
    # print('alpha=%0.3f' % (0.05/len(dists)))
    ax.legend(loc=4,fontsize=8)

    return ax, anova_df#fig, ax

def plot_variable_condition(df_cond,condition,variable,x_min,x_max,y_min,y_max,color_scheme,save_pdf,pp):
    ###currently set to success trials only
    conds = np.unique(df_cond[condition])
    anis = np.unique(df_cond['subject'])

    d_var = np.zeros((len(anis),len(conds)))
    for c,cond in enumerate(conds):
        df = df_cond[(df_cond[condition]==cond)&(df_cond['success']==1)]
        jumpcurve = df.groupby(['subject']).median()
        d_var[:,c] = jumpcurve[variable].to_numpy()

    fig, axs = plt.subplots(1,1,figsize=(2,2))
    for a,ani in enumerate(anis):
        axs.plot([1,2],[d_var[a,0],d_var[a,1]],'-',linewidth=0.25,alpha=0.25,color=[0.5,0.5,0.5])
    axs.scatter(np.ones(d_var.shape[0]),d_var[:,0],marker='o',alpha=0.25,color=color_scheme[0],edgecolors='none')
    axs.scatter(np.ones(d_var.shape[0])+1,d_var[:,1],marker='o',alpha=0.25,color=color_scheme[1],edgecolors='none')
    axs.errorbar(x=[1,2],y=[np.mean(d_var[:,0]),np.mean(d_var[:,1])],yerr=[np.std(d_var[:,0])/np.sqrt(len(anis)),np.std(d_var[:,1])/np.sqrt(len(anis))],color='k',linewidth=1,zorder=0)
    axs.set_ylim(y_min,y_max)
    axs.set_xlim(x_min,x_max)
    axs.set_ylabel(variable)
    axs.set_xticks([1,len(conds)])
    axs.set_xticklabels([conds[0],conds[1]])
    axs = xy_axis(axs)
    fig.tight_layout()
    if save_pdf:
        pp.savefig(fig)
        plt.close(fig)

    s,p = stats.ttest_rel(d_var[:,0],d_var[:,1])
    print('%s %0.2f+/-%0.2f, %s %0.2f+/-%0.2f, %s %s t-test p=%0.3f' % (conds[0],np.mean(d_var[:,0]),np.std(d_var[:,0])/np.sqrt(len(anis)),\
        conds[1],np.mean(d_var[:,1]),np.std(d_var[:,1])/np.sqrt(len(anis)),condition,variable,p))

    if save_pdf:
        pp.savefig(fig)
        plt.close(fig)

    return fig, axs, p, d_var




def plot_performance_platforms(axs,og_df,condition,aborts,plt_min,plt_max,color_scheme,ls_list):#,save_pdf,pp):
    
    ### Plot success versus distance for two conditions as a function of the platforms
    
    ### INPUTS
    ### df: the dataframe containing all of the data
    ### condition: the key from df that you want to split the data into (e.g. 'ocular')
    ### plot_params: a dictionary of parameters including a colormap 'cm' and colors 'cond_cols'
    ### save_pdf: whether option was selected to save pdf
    ### pp: the pdf object (only necessary if saving PDF)
    
    ### OUTPUTS
    ### fig, ax: the figure and axes objects
    if aborts:
        df = aborts_as_failures(og_df)
    else:
        df = remove_aborts(og_df)
    # df = aborts_as_failures(og_df)
    anis = np.unique(df['subject'])
    dists = np.unique(df['distance_DLC'])
    conds = np.unique(df[condition])
    
    # fig, axs = plt.subplots(1,len(conds),figsize=(5*len(conds),5))
    if len(conds)>1:
        axs = axs.ravel()
    else:
        ax = axs

    for c,cond in enumerate(conds):
        anova_df = pd.DataFrame(columns=['distance_DLC','platform','success'])
        if len(conds)>1:
            ax = axs[c]
        for pl in range(1,4):
            jumpcurve = df[(df['platform_DLC']==pl)&(df[condition]==cond)].groupby(['distance_DLC','subject']).mean()
            # jumpcurve.reset_index(inplace=True)
            mnplot = jumpcurve.groupby(['distance_DLC']).mean()
            semplot = jumpcurve.groupby(['distance_DLC']).std()/np.sqrt(len(anis))
            mnplot.reset_index(inplace=True)
            semplot.reset_index(inplace=True)
            ax.errorbar(mnplot['distance_DLC'],
                         mnplot['success'],
                         yerr=semplot['success'],color=color_scheme[c],ls=ls_list[pl-1],label='platform %d' % pl,linewidth=1)
            for d,dist in enumerate(dists):
                temp_df = pd.DataFrame(columns=['distance_DLC','platform_DLC','jumpdist'])
                temp_df['distance_DLC']  = pd.Series(np.repeat(dist,len(anis)))
                temp_df['platform_DLC'] = pd.Series(np.repeat(pl,len(anis)))
                temp_df['success'] = pd.Series(jumpcurve.xs(dist)['success'].to_numpy())
                anova_df = pd.concat([anova_df,temp_df],axis=0)

        ax.set_xlabel('gap distance (cm)')
        ax.set_ylabel('success rate')
        ax.set_xlim(plt_min,plt_max)
        # ax.set_xticks([10,15,20,25])
        # ax.set_xticks(np.arange(plt_min,plt_max+plt_min,plt_min))
        ax.set_ylim(0,1.1)
        
        # ax.set_title(cond)
        ax.set_xticks(np.arange(10, 30, step=5))
        ax.set_yticks(np.arange(0, 1.5, step=0.5))
        ax = xy_axis(ax)

        # #print anova results
        model = ols('success ~ C(distance_DLC) + C(platform_DLC) + C(distance_DLC):C(platform_DLC)', data=anova_df).fit()
        print(cond)
        print(sm.stats.anova_lm(model, typ=2))
        print('')
    # if len(anis)==1:
    #     fig.suptitle(anis[0])
    # # fig.tight_layout()

    # if save_pdf:
    #     pp.savefig(fig)
    #     plt.close(fig)
    ax.legend(fontsize=8,loc=3)

    return ax#fig, ax


def plot_jumpdist_platforms(axs,og_df,condition,plt_min,plt_max,color_scheme,ls_list):#,save_pdf,pp):
    
    ### Plot success versus distance for two conditions as a function of the platforms
    
    ### INPUTS
    ### df: the dataframe containing all of the data
    ### condition: the key from df that you want to split the data into (e.g. 'ocular')
    ### plot_params: a dictionary of parameters including a colormap 'cm' and colors 'cond_cols'
    ### save_pdf: whether option was selected to save pdf
    ### pp: the pdf object (only necessary if saving PDF)
    
    ### OUTPUTS
    ### fig, ax: the figure and axes objects
    df = remove_aborts(og_df)
    anis = np.unique(df['subject'])
    dists = np.unique(df['distance_DLC'])
    conds = np.unique(df[condition])

    # fig, axs = plt.subplots(1,len(conds),figsize=(5*len(conds),5))
    if len(conds)>1:
        axs = axs.ravel()
    else:
        ax = axs
        
    for c,cond in enumerate(conds):
        anova_df = pd.DataFrame(columns=['distance_DLC','platform_DLC','jumpdist'])
        if len(conds)>1:
            ax = axs[c]
        for pl in range(1,4):
            jumpcurve = df[(df['platform_DLC']==pl)&(df['success']==1)&(df[condition]==cond)].groupby(['distance_DLC','subject']).mean()
            # jumpcurve.reset_index(inplace=True)
            mnplot = jumpcurve.groupby(['distance_DLC']).mean()
            semplot = jumpcurve.groupby(['distance_DLC']).std()/np.sqrt(len(anis))
            mnplot.reset_index(inplace=True)
            semplot.reset_index(inplace=True)
            semplot[np.isnan(semplot)]=0
            xvals = np.unique(mnplot['distance_DLC'])
            ax.errorbar(xvals,
                         mnplot['jumpdist'],
                         yerr=semplot['jumpdist'],label='platform %d' % pl,color=color_scheme[c],ls=ls_list[pl-1],linewidth=1)
            for d,dist in enumerate(dists):
                temp_df = pd.DataFrame(columns=['distance_DLC','platform_DLC','jumpdist'])
                temp_df['distance_DLC']  = pd.Series(np.repeat(dist,len(anis)))
                temp_df['platform_DLC'] = pd.Series(np.repeat(pl,len(anis)))
                temp_df['jumpdist'] = pd.Series(jumpcurve.xs(dist)['jumpdist'].to_numpy())
                anova_df = pd.concat([anova_df,temp_df],axis=0)
        ax.set_xlabel('gap distance (cm)')
        ax.set_ylabel('distance jumped (cm)')
        ax.set_xlim(plt_min,plt_max)
        ax.set_ylim(plt_min,plt_max)
        ax.set_xticks([10,15,20,25])
        ax.set_yticks([10,15,20,25])
        # ax.set_xticks(np.arange(plt_min,plt_max+plt_min,plt_min))
        # ax.set_yticks(np.arange(plt_min,plt_max+plt_min,plt_min))
        # ax1.set_title('both bi/monocular')
        # locs, labels = plt.xticks(fontsize=10)
        # plt.xticks(np.arange(8, 32, step=4))
        # for pl in range(3):
        ax.plot(xvals,xvals,'k:')
        # ax.set_title(cond)
        ax = xy_axis(ax)

        try:
            #print anova results
            model = ols('jumpdist ~ C(distance_DLC) + C(platform_DLC) + C(distance_DLC):C(platform_DLC)', data=anova_df).fit()
            print(cond)
            print(sm.stats.anova_lm(model, typ=2))
            print('')
        except:
            print('could not do anova bc nans')

    # if len(anis)==1:
    #     fig.suptitle(anis[0])
    # fig.tight_layout()

    # if save_pdf:
    #     pp.savefig(fig)
    #     plt.close(fig)
    ax.legend(fontsize=8,loc=4)  

    return axs#fig, axs



def plot_ris_distance(og_df,condition,ris,plot_params,save_pdf,pp):
    
    ### Plot success versus distance for two conditions as a function of the platforms
    
    ### INPUTS
    ### df: the dataframe containing all of the data
    ### condition: the key from df that you want to split the data into (e.g. 'ocular')
    ### ris: table of retinal image sizes generated by make_ris_table
    ### plot_params: a dictionary of parameters including a colormap 'cm' and colors 'cond_cols'
    ### save_pdf: whether option was selected to save pdf
    ### pp: the pdf object (only necessary if saving PDF)
    
    ### OUTPUTS
    ### fig, ax: the figure and axes objects
    df = remove_aborts(og_df)
    anis = np.unique(df['subject'])
    dists = np.unique(df['distance_DLC'])
    conds = np.unique(df[condition])

    fig, axs = plt.subplots(1,len(conds),figsize=(5*len(conds),5))
    if len(conds)>1:
        axs = axs.ravel()
    else:
        ax = axs
    
    for c,cond in enumerate(conds):
        if len(conds)>1:
            ax = axs[c]
        for pl in range(1,4):
            xvals = ris[pl-1,:]
            jumpcurve = df[(df['platform']==pl)&(df['success']==1)&(df[condition]==cond)].groupby(['subject','distance_DLC','jumpdist']).mean()
            jumpcurve.reset_index(inplace=True)
            mnplot = jumpcurve.groupby(['distance_DLC']).mean()
            semplot = jumpcurve.groupby(['distance_DLC']).std()/np.sqrt(len(anis))
            mnplot.reset_index(inplace=True)
            semplot.reset_index(inplace=True)
            ax.errorbar(xvals,
                         mnplot['jumpdist'],
                         yerr=semplot['jumpdist'],color=plot_params['plat_cols'][pl-1],label='platform %d' % pl)
            ax.plot(ris[pl-1,:],np.unique(mnplot['distance_DLC']),':',color=plot_params['plat_cols'][pl-1])
        ax.set_xlabel('retinal image size (deg)')
        ax.set_ylabel('distance jumped (cm)')
        ax.set_xlim(np.round(np.min(ris),decimals=-1),np.round(np.max(ris),decimals=-1))
        # ax1.set_ylim(0,1.1)
        ax.legend(fontsize=10)
        ax = xy_axis(ax)
        # ax1.set_title('both bi/monocular')
        # locs, labels = plt.xticks()
        # plt.xticks(np.arange(8, 32, step=4))
    if len(anis)==1:
        fig.suptitle(anis[0])
    if save_pdf:
        pp.savefig(fig)
        plt.close(fig)
        
    return fig, ax



def plot_variable_vs_distance(ax,df,variable,condition,x_min,x_max,y_min,y_max,color_scheme,save_pdf,pp,suptitle=''):
    
    ### Plot success versus distance for two conditions
    
    ### INPUTS
    ### df: the dataframe containing all of the data
    ### condition: the key from df that you want to split the data into (e.g. 'ocular')
    ### plot_params: a dictionary of parameters including a colormap 'cm' and colors 'cond_cols'
    ### save_pdf: whether option was selected to save pdf
    ### pp: the pdf object (only necessary if saving PDF)
    
    ### OUTPUTS
    ### fig, ax: the figure and axes objects
    anis = np.unique(df['subject'])
    dists = np.unique(df['distance_DLC'])
    conds = np.unique(df[condition])
    # suc_lab = ['failure','success']

    # fig, axs = plt.subplots(1,len(conds),figsize=(2*len(conds),2))
    # axs = axs.ravel()
    # y_max=[]
    for c,cond in enumerate(conds):
        # stats_array = np.empty((len(anis),len(dists),2))
        # stats_array[:] = np.nan
        # anova_df = pd.DataFrame(columns=['distance_DLC','success',variable])

        # for suc in range(2):
        # jumpcurve = df[(df['success']==suc)&(df[condition]==cond)].groupby(['distance_DLC','subject']).mean()
        jumpcurve = df[(df[condition]==cond)].groupby(['distance_DLC','subject']).mean()
        # jumpcurve.reset_index(inplace=True)
        mnplot = jumpcurve.groupby(['distance_DLC']).mean()
        semplot = jumpcurve.groupby(['distance_DLC']).std()/np.sqrt(len(anis))
        mnplot.reset_index(inplace=True)
        semplot.reset_index(inplace=True)
        semplot[np.isnan(semplot)]=0
        xvals = np.unique(mnplot['distance_DLC'])
        # ax = axs[c]
        ax.errorbar(xvals,
                        mnplot[variable],
                        yerr=semplot[variable],label=cond,color=color_scheme[c],linewidth=1,zorder=c)
        jumpcurve.reset_index(inplace=True)            
        for ani in anis:
            ax.plot(jumpcurve[jumpcurve['subject']==ani]['distance_DLC'],jumpcurve[jumpcurve['subject']==ani][variable],'-',color=color_scheme[c],linewidth=0.25,alpha=0.5)
        ax.set_xlabel('gap distance (cm)')
        ax.set_ylabel(variable)
        ax.axis([x_min,x_max,y_min,y_max])
        ax.set_xticks(np.arange(10,30,5))
        # ax.set_ylim(plt_min,plt_max)
        # ax.set_yticks(np.arange(plt_min,plt_max+plt_max/4,plt_max/4))
        # y_max.append(np.ceil(np.max(mnplot[variable])))
        # ax.set_title(cond)
        # ax1.set_title('both bi/monocular')
        # locs, labels = plt.xticks(fontsize=10)
        # plt.xticks(np.arange(8, 32, step=4))
        # for pl in range(3):
        # ax.plot(xvals,xvals,'k:')
        ax = xy_axis(ax)

        #     for d,dist in enumerate(dists):
        #         try:
        #             stats_array[:,d,suc] = jumpcurve.xs(dist)[variable]
        #             temp_df = pd.DataFrame(columns=['distance_DLC','success',variable])
        #             temp_df['distance_DLC']  = pd.Series(np.repeat(dist,len(anis)))
        #             temp_df['success'] = pd.Series(np.repeat(suc,len(anis)))
        #             temp_df[variable] = pd.Series(jumpcurve.xs(dist)[variable].to_numpy())
        #             anova_df = pd.concat([anova_df,temp_df],axis=0)
        #         except:
        #             stats_array[:,d,suc] = np.repeat(np.nan,len(anis))
        #             temp_df = pd.DataFrame(columns=['distance_DLC','success',variable])
        #             temp_df['distance_DLC']  = pd.Series(np.repeat(dist,len(anis)))
        #             temp_df['success'] = pd.Series(np.repeat(suc,len(anis)))
        #             temp_df[variable] = pd.Series(np.repeat(np.nan,len(anis)))
        #             anova_df = pd.concat([anova_df,temp_df],axis=0)

        # #print anova results
        # try:
        #     model = ols('%s ~ C(distance) + C(condition) + C(distance):C(condition)' % variable, data=anova_df).fit()
        #     print(cond)
        #     print(sm.stats.anova_lm(model, typ=2))
        #     print('')
        # except:
        #     print('could not do anova due to nans')

        # for d,dist in enumerate(dists):
        #     fail = stats_array[:,d,0]
        #     suc = stats_array[:,d,1]
        #     fail = fail[~np.isnan(fail)]
        #     suc = suc[~np.isnan(suc)]
        #     s, p = stats.ttest_ind(fail,suc)
        #     print('distance %d, p=%0.3f' % (dist,p))
        # print('alpha=%0.3f' % (0.05/len(dists)))


    # if len(anis)==1:
    #     fig.suptitle(anis[0])
    # ax.legend(fontsize=10)
    # fig.tight_layout()
    # y_max = np.max(y_max)
    # y_min = 0
    # for c in range(len(conds)):
    #     axs[c].set_ylim(plt_min,plt_max)
    #     axs[c].set_yticks(np.arange(plt_min,plt_max+plt_max/4,plt_max/4))
        # axs[c].set_ylim(y_min,y_max)
        # axs[c].set_yticks(np.arange(y_min,y_max+y_max/4,y_max/4))

    # fig.suptitle(suptitle)
    # if save_pdf:
    #     pp.savefig(fig)
    #     plt.close(fig)
        
    return ax#fig, ax


def plot_variable_vs_distance_manipulation(axs,df,variable,condition,manipulation,x_min,x_max,y_min,y_max,color_scheme,save_pdf,pp,suptitle=''):
    
    ### Plot success versus distance for two conditions
    
    ### INPUTS
    ### df: the dataframe containing all of the data
    ### condition: the key from df that you want to split the data into (e.g. 'ocular')
    ### plot_params: a dictionary of parameters including a colormap 'cm' and colors 'cond_cols'
    ### save_pdf: whether option was selected to save pdf
    ### pp: the pdf object (only necessary if saving PDF)
    
    ### OUTPUTS
    ### fig, ax: the figure and axes objects
    anis = np.unique(df['subject'])
    dists = np.unique(df['distance_DLC'])
    conds = np.unique(df[condition])
    mans = np.unique(df[manipulation])

    axs = axs.ravel()
    # y_max=[]
    for c,cond in enumerate(conds):
        # stats_array = np.empty((len(anis),len(dists),2))
        # stats_array[:] = np.nan
        # anova_df = pd.DataFrame(columns=['distance_DLC','success',variable])
        ax = axs[c]
        for m,man in enumerate(mans):
            jumpcurve = df[(df[manipulation]==man)&(df[condition]==cond)].groupby(['distance_DLC','subject']).mean()
            # jumpcurve.reset_index(inplace=True)
            mnplot = jumpcurve.groupby(['distance_DLC']).mean()
            semplot = jumpcurve.groupby(['distance_DLC']).std()/np.sqrt(len(anis))
            mnplot.reset_index(inplace=True)
            semplot.reset_index(inplace=True)
            semplot[np.isnan(semplot)]=0
            xvals = np.unique(mnplot['distance_DLC'])
            
            ax.errorbar(xvals,
                         mnplot[variable],
                         yerr=semplot[variable],label='%s %s' % (manipulation,man),color=color_scheme[m],linewidth=1,zorder=c)
            jumpcurve.reset_index(inplace=True)            
            for ani in anis:
                ax.plot(jumpcurve[jumpcurve['subject']==ani]['distance_DLC'],jumpcurve[jumpcurve['subject']==ani][variable],':',color=color_scheme[m],linewidth=0.25,alpha=0.5)
        ax.set_xlabel('gap distance (cm)')
        ax.set_ylabel(variable)
        ax.set_xlim(x_min,x_max)
        ax.set_xticks(np.arange(10,30,5))
        ax.set_ylim(y_min,y_max)
        # ax.set_yticks(np.arange(plt_min,plt_max+plt_max/4,plt_max/4))
        # y_max.append(np.ceil(np.max(mnplot[variable])))
        # ax.set_title(cond)
        # ax1.set_title('both bi/monocular')
        # locs, labels = plt.xticks(fontsize=10)
        # plt.xticks(np.arange(8, 32, step=4))
        # for pl in range(3):
        # ax.plot(xvals,xvals,'k:')
        ax = xy_axis(ax)

        #     for d,dist in enumerate(dists):
        #         try:
        #             stats_array[:,d,suc] = jumpcurve.xs(dist)[variable]
        #             temp_df = pd.DataFrame(columns=['distance_DLC','success',variable])
        #             temp_df['distance_DLC']  = pd.Series(np.repeat(dist,len(anis)))
        #             temp_df['success'] = pd.Series(np.repeat(suc,len(anis)))
        #             temp_df[variable] = pd.Series(jumpcurve.xs(dist)[variable].to_numpy())
        #             anova_df = pd.concat([anova_df,temp_df],axis=0)
        #         except:
        #             stats_array[:,d,suc] = np.repeat(np.nan,len(anis))
        #             temp_df = pd.DataFrame(columns=['distance_DLC','success',variable])
        #             temp_df['distance_DLC']  = pd.Series(np.repeat(dist,len(anis)))
        #             temp_df['success'] = pd.Series(np.repeat(suc,len(anis)))
        #             temp_df[variable] = pd.Series(np.repeat(np.nan,len(anis)))
        #             anova_df = pd.concat([anova_df,temp_df],axis=0)

        # #print anova results
        # try:
        #     model = ols('%s ~ C(distance) + C(condition) + C(distance):C(condition)' % variable, data=anova_df).fit()
        #     print(cond)
        #     print(sm.stats.anova_lm(model, typ=2))
        #     print('')
        # except:
        #     print('could not do anova due to nans')

        # for d,dist in enumerate(dists):
        #     fail = stats_array[:,d,0]
        #     suc = stats_array[:,d,1]
        #     fail = fail[~np.isnan(fail)]
        #     suc = suc[~np.isnan(suc)]
        #     s, p = stats.ttest_ind(fail,suc)
        #     print('distance %d, p=%0.3f' % (dist,p))
        # print('alpha=%0.3f' % (0.05/len(dists)))


    # if len(anis)==1:
    #     fig.suptitle(anis[0])
    # ax.legend(fontsize=8)
    # fig.tight_layout()
    
    # y_max = np.max(y_max)
    # y_min = 0
    # for c in range(len(conds)):
    #     axs[c].set_ylim(plt_min,plt_max)
    #     axs[c].set_yticks(np.arange(plt_min,plt_max+plt_max/4,plt_max/4))
    #     # axs[c].set_ylim(y_min,y_max)
    #     # axs[c].set_yticks(np.arange(y_min,y_max+y_max/4,y_max/4))

    # fig.suptitle(suptitle)
    # if save_pdf:
    #     pp.savefig(fig)
    #     plt.close(fig)
        
    return axs#fig, ax




def plot_variable_manipulation(df,variable,condition,manipulation,xvals,plt_min,plt_max,plot_params,save_pdf,pp,suptitle=''):
    
    ### Plot success versus distance for two conditions
    
    ### INPUTS
    ### df: the dataframe containing all of the data
    ### condition: the key from df that you want to split the data into (e.g. 'ocular')
    ### plot_params: a dictionary of parameters including a colormap 'cm' and colors 'cond_cols'
    ### save_pdf: whether option was selected to save pdf
    ### pp: the pdf object (only necessary if saving PDF)
    
    ### OUTPUTS
    ### fig, ax: the figure and axes objects
    anis = np.unique(df['subject'])
    conds = np.unique(df[condition])
    mans = np.unique(df[manipulation])

    fig, axs = plt.subplots(1,len(conds),figsize=(10,5))
    axs = axs.ravel()
    y_max=[]
    for c,cond in enumerate(conds):
        # stats_array = np.empty((len(anis),len(dists),2))
        # stats_array[:] = np.nan
        # anova_df = pd.DataFrame(columns=['distance_DLC','success',variable])

        for m,man in enumerate(mans):
            jumpcurve = df[(df[manipulation]==man)&(df[condition]==cond)].groupby(['distance_DLC','subject']).apply(np.mean)
            # jumpcurve.reset_index(inplace=True)
            mnplot = jumpcurve.groupby(['distance_DLC']).apply(np.mean)
            semplot = jumpcurve.groupby(['distance_DLC']).apply(np.std()/np.sqrt(len(anis)))
            mnplot.reset_index(inplace=True)
            semplot.reset_index(inplace=True)
            semplot[np.isnan(semplot)]=0
            ax = axs[c]
            ax.errorbar(xvals,
                         mnplot[variable],
                         yerr=semplot[variable],label='%s %s' % (manipulation,man),color=plot_params['laser_col'][m])
            jumpcurve.reset_index(inplace=True)            
            for ani in anis:
                ax.plot(xvals,jumpcurve[jumpcurve['subject']==ani][variable],'-',color=plot_params['laser_col'][m],linewidth=0.4)
            ax.set_xlabel('gap distance (cm)')
            ax.set_ylabel(variable)
            ax.set_xlim(5,25)
            ax.set_xticks(np.arange(5,30,5))
            # ax.set_ylim(plt_min,plt_max)
            # ax.set_yticks(np.arange(plt_min,plt_max+plt_max/4,plt_max/4))
            y_max.append(np.ceil(np.max(mnplot[variable])))
            ax.set_title(cond)
            # ax1.set_title('both bi/monocular')
            # locs, labels = plt.xticks(fontsize=10)
            # plt.xticks(np.arange(8, 32, step=4))
            # for pl in range(3):
            # ax.plot(xvals,xvals,'k:')
            ax = xy_axis(ax)

        #     for d,dist in enumerate(dists):
        #         try:
        #             stats_array[:,d,suc] = jumpcurve.xs(dist)[variable]
        #             temp_df = pd.DataFrame(columns=['distance_DLC','success',variable])
        #             temp_df['distance_DLC']  = pd.Series(np.repeat(dist,len(anis)))
        #             temp_df['success'] = pd.Series(np.repeat(suc,len(anis)))
        #             temp_df[variable] = pd.Series(jumpcurve.xs(dist)[variable].to_numpy())
        #             anova_df = pd.concat([anova_df,temp_df],axis=0)
        #         except:
        #             stats_array[:,d,suc] = np.repeat(np.nan,len(anis))
        #             temp_df = pd.DataFrame(columns=['distance_DLC','success',variable])
        #             temp_df['distance_DLC']  = pd.Series(np.repeat(dist,len(anis)))
        #             temp_df['success'] = pd.Series(np.repeat(suc,len(anis)))
        #             temp_df[variable] = pd.Series(np.repeat(np.nan,len(anis)))
        #             anova_df = pd.concat([anova_df,temp_df],axis=0)

        # #print anova results
        # try:
        #     model = ols('%s ~ C(distance) + C(condition) + C(distance):C(condition)' % variable, data=anova_df).fit()
        #     print(cond)
        #     print(sm.stats.anova_lm(model, typ=2))
        #     print('')
        # except:
        #     print('could not do anova due to nans')

        # for d,dist in enumerate(dists):
        #     fail = stats_array[:,d,0]
        #     suc = stats_array[:,d,1]
        #     fail = fail[~np.isnan(fail)]
        #     suc = suc[~np.isnan(suc)]
        #     s, p = stats.ttest_ind(fail,suc)
        #     print('distance %d, p=%0.3f' % (dist,p))
        # print('alpha=%0.3f' % (0.05/len(dists)))


    # if len(anis)==1:
    #     fig.suptitle(anis[0])
    # fig.tight_layout()
    ax.legend(fontsize=10)
    y_max = np.max(y_max)
    y_min = 0
    for c in range(len(conds)):
        axs[c].set_ylim(plt_min,plt_max)
        axs[c].set_yticks(np.arange(plt_min,plt_max+plt_max/4,plt_max/4))
        # axs[c].set_ylim(y_min,y_max)
        # axs[c].set_yticks(np.arange(y_min,y_max+y_max/4,y_max/4))

    fig.suptitle(suptitle)
    if save_pdf:
        pp.savefig(fig)
        plt.close(fig)
        
    return fig, ax




# calculate the 'accuracy' by getting the absolute value of the difference between expected (mean) and actual jump distance
def get_jump_accuracy(df):
    anis = np.unique(df['subject'])
    dists = np.unique(df['distance_DLC'])
    mn_jumpdist = np.zeros((len(anis),len(dists)))
    df2 = df[['subject','distance_DLC','success','jumpdist']][df['success']==1].copy()
    df2 = df2.groupby(['subject','success','distance_DLC']).mean().unstack(fill_value=np.nan).stack(dropna=False)
    for a,ani in enumerate(anis):
        mn_jumpdist[a,:] = df2.xs([ani],level=[0])['jumpdist'].to_numpy()
    
    df['mn_jumpdist'] = np.zeros((df.shape[0]))
    for index, row in df.iterrows():
        df['mn_jumpdist'].iloc[index] = mn_jumpdist[np.where(anis==row['subject'])[0][0],np.where(dists==row['distance_DLC'])[0][0]]


    df['accuracy'] = np.abs(df['mn_jumpdist']-df['jumpdist'])

    return df


def label_bob_trials(df,labels,cluster_key,cluster_num):
#create a column in labels corresponding to whether a trial contains a given movement
    bob_trial = np.zeros((df.shape[0],)).astype('int')
    df['bob_trial'] = bob_trial

    for idx,row in df.iterrows():
        if labels[(labels['subject']==row['subject'])&(labels['expdate']==row['expdate'])& \
        (labels['trial']==row['trial'])&(labels[cluster_key]==cluster_num)]['trial'].empty:
            pass
        else:
            df.loc[idx,'bob_trial']=1

    return df

def label_any_bob_trials(df,labels,cluster_key):
#create a column in labels corresponding to whether a trial contains a given movement
    bob_trial = np.zeros((df.shape[0],)).astype('int')
    df['bob_trial'] = bob_trial

    for idx,row in df.iterrows():
        if labels[(labels['subject']==row['subject'])&(labels['expdate']==row['expdate'])& \
        (labels['trial']==row['trial'])]['trial'].empty:
            pass
        else:
            df.loc[idx,'bob_trial']=1

    return df

def label_bob_frequency(df,labels,cluster_key,cluster_num):
#create a column in labels corresponding to whether a trial contains a given movement
    bob_trial = np.zeros((df.shape[0],)).astype('int')
    df['bob_freq'] = bob_trial

    for idx,row in df.iterrows():
        freq = len(labels[(labels['subject']==row['subject'])&(labels['expdate']==row['expdate'])& \
        (labels['trial']==row['trial'])&(labels[cluster_key]==cluster_num)]['trial'])

        df.loc[idx,'bob_freq']=freq

    return df


def plot_performance_bob_freq(df,condition,bob_freq_label,freq_max,save_pdf,pp,suptitle):
    ### Plot performance as a function of bob frequency

    ### INPUTS
    ### df: the dataframe with all of the data
    ### condition: the key from df that you want to split the data into (e.g. 'ocular')
    ### bob_freq_label: the column in df containing the bob frequency of interest, e.g. 'bob_freq'
    ### freq_max: the maximum number of bobs to plot a line for (from 0 to freq_max)
    ### plot_params: dictionary containing parameters for plotting
    ### save_pdf: whether option was selected to save pdf
    ### pp: the pdf object (only necessary if saving PDF)

    ### OUTPUTS
    ### fig, ax: the figure and axes objects

    anis = np.unique(df['subject'])
    dists = np.unique(df['distance_DLC'])
    conds = np.unique(df[condition])
    suc_lab = ['fail','success']
    suc_lim = [10,3]
    jet = cm.get_cmap('jet',freq_max+1)
    new_cols = jet(np.linspace(0, 1, freq_max+1))

    fig, axs = plt.subplots(4,len(conds),figsize=(len(conds)*5,5*4))
    temp_df = remove_aborts(df)
    temp_df = temp_df[['subject','distance_DLC','success','jumpdist','accuracy',condition,bob_freq_label]]

    for c,cond in enumerate(conds):

        anova_df = pd.DataFrame(columns=['distance_DLC','freq','success'])

        for freq in np.arange(freq_max+1):
            
            if freq==0:
                mn = temp_df[(temp_df[bob_freq_label]==freq)&(temp_df[condition]==cond)].groupby(['subject','distance_DLC']).mean()
                mn.reset_index(inplace=True)
                tot_tr = len(temp_df[(temp_df[bob_freq_label]==freq)&(temp_df[condition]==cond)])
            else:
                mn = temp_df[(temp_df[bob_freq_label]>=freq)&(temp_df[condition]==cond)].groupby(['subject','distance_DLC']).mean()
                mn.reset_index(inplace=True)
                tot_tr = len(temp_df[(temp_df[bob_freq_label]>=freq)&(temp_df[condition]==cond)])
            
            try:
                for d,dist in enumerate(dists):
    #                 stats_array[:,d,freq] = mn.xs(dist)['success']
                    stats_df = pd.DataFrame(columns=['distance_DLC','freq','success'])
                    stats_df['distance_DLC']  = pd.Series(np.repeat(dist,len(anis)))
                    stats_df['freq'] = pd.Series(np.repeat(freq,len(anis)))
                    stats_df['success'] = pd.Series(mn.xs(dist)['success'].to_numpy())
                    anova_df = pd.concat([anova_df,stats_df],axis=0)
            except:
                pass
             
            if len(mn)>0:
                mnplot = mn.groupby(['distance_DLC']).mean()
                mnplot.reset_index(inplace=True)
                semplot = mn.groupby(['distance_DLC']).std()/np.sqrt(len(anis))
                semplot.reset_index(inplace=True)
                if freq==0:
                    axs[0,c].errorbar(mnplot['distance_DLC'],
                                    mnplot['success'],
                                    yerr=semplot['success'],
                                    label='%d n=%d' % (freq,tot_tr),
                                    color=new_cols[freq]) #color=plot_params['hmm_col'][0] #this is for 0 vs. 1+
                else:
                    axs[0,c].errorbar(mnplot['distance_DLC'],
                                    mnplot['success'],
                                    yerr=semplot['success'],
                                    label='%d+ n=%d' % (freq,tot_tr),
                                    color=new_cols[freq])
            if freq==freq_max:
                try:
                    model = ols('success ~ C(distance_DLC) + C(freq) + C(distance):C(freq)', data=anova_df).fit()
                    anova_stats = sm.stats.anova_lm(model, typ=2)
                    pval = 'p=%0.3f' % anova_stats['PR(>F)'].loc['C(freq)']
                except:
                    pval = 'p=NaN'
                axs[0,c].plot(0,0,'ko',markersize=0,label='anova %s' % pval)

            for suc in range(2):
                ax = axs[suc+1,c]
                if freq==0:
                    mn = temp_df[(temp_df[bob_freq_label]==freq)&(temp_df[condition]==cond)&(temp_df['success']==suc)].groupby(['subject','distance_DLC']).mean()
                    tot_tr = len(temp_df[(temp_df[bob_freq_label]==freq)&(temp_df[condition]==cond)&(temp_df['success']==suc)])
                else:
                    mn = temp_df[(temp_df[bob_freq_label]>=freq)&(temp_df[condition]==cond)&(temp_df['success']==suc)].groupby(['subject','distance_DLC']).mean()
                    tot_tr = len(temp_df[(temp_df[bob_freq_label]>=freq)&(temp_df[condition]==cond)&(temp_df['success']==suc)])
                mn.reset_index(inplace=True)
                if len(mn)>0:
                    mnplot = mn.groupby(['distance_DLC']).mean()
                    mnplot.reset_index(inplace=True)
                    semplot = mn.groupby(['distance_DLC']).mean()/np.sqrt(len(anis))
                    semplot.reset_index(inplace=True)
                    if freq==0:
                        ax.errorbar(mnplot['distance_DLC'],
                                        mnplot['accuracy'],
                                        yerr=semplot['accuracy'],
                                        label='%d n=%d' % (freq,tot_tr),
                                        color=new_cols[freq])
                    else:
                        ax.errorbar(mnplot['distance_DLC'],
                                        mnplot['accuracy'],
                                        yerr=semplot['accuracy'],
                                        label='%d+ n=%d' % (freq,tot_tr),
                                        color=new_cols[freq])

            ax = axs[3,c]
            if freq==0:
                mn = temp_df[(temp_df[bob_freq_label]==freq)&(temp_df[condition]==cond)].groupby(['subject','distance_DLC']).mean()
                tot_tr = len(temp_df[(temp_df[bob_freq_label]==freq)&(temp_df[condition]==cond)])
            else:
                mn = temp_df[(temp_df[bob_freq_label]>=freq)&(temp_df[condition]==cond)].groupby(['subject','distance_DLC']).mean()
                tot_tr = len(temp_df[(temp_df[bob_freq_label]>=freq)&(temp_df[condition]==cond)])
            mn.reset_index(inplace=True)
            if len(mn)>0:
                mnplot = mn.groupby(['distance_DLC']).mean()
                mnplot.reset_index(inplace=True)
                semplot = mn.groupby(['distance_DLC']).mean()/np.sqrt(len(anis))
                semplot.reset_index(inplace=True)
                if freq==0:
                    ax.errorbar(mnplot['distance_DLC'],
                                    mnplot['accuracy'],
                                    yerr=semplot['accuracy'],
                                    label='%d n=%d' % (freq,tot_tr),
                                    color=new_cols[freq])
                else:
                    ax.errorbar(mnplot['distance_DLC'],
                                    mnplot['accuracy'],
                                    yerr=semplot['accuracy'],
                                    label='%d+ n=%d' % (freq,tot_tr),
                                    color=new_cols[freq])
                
        ax = axs[0,c]
        ax.set_xlabel('gap distance (cm)')
        ax.set_ylabel('success rate')
        ax.set_title(('%s') % cond)
        ax.set_xlim(5,25)
        ax.set_ylim(0,1.1)
        ax.set_yticks([0,0.5,1.0])
        ax.legend(fontsize=10)
        ax.set_xticks(np.arange(5, 30, step=5))
        ax = xy_axis(ax)

        for suc in range(2):
            ax = axs[suc+1,c]
            ax.set_xlabel('gap distance (cm)')
            ax.set_ylabel('error (cm)')
            ax.set_title(suc_lab[suc])
            ax.set_xlim(5,25)
            ax.set_ylim(0,suc_lim[suc])
            ax.set_yticks(np.arange(0,suc_lim[suc]+1,1))
            ax.legend(fontsize=10)
            ax.set_xticks(np.arange(10, 30, step=5))
            ax = xy_axis(ax)

        ax = axs[3,c]
        ax.set_xlabel('gap distance (cm)')
        ax.set_ylabel('error (cm)')
        ax.set_title('all trials')
        ax.set_xlim(5,25)
        ax.set_ylim(0,suc_lim[0])
        ax.set_yticks(np.arange(0,suc_lim[0]+1,1))
        ax.legend(fontsize=10)
        ax.set_xticks(np.arange(10, 30, step=5))
        ax = xy_axis(ax)


        

    fig.suptitle(suptitle)
    fig.tight_layout()

    # plt.tight_layout()

    if save_pdf:
        pp.savefig(fig)
        plt.close(fig)

    return fig, ax


def plot_state_freq_difference(df,st_remap,plt_types,plt_lims,time_per,condition,save_pdf,pp):
    dists = np.unique(df['distance_DLC'])
    conds = np.unique(df[condition])
    anis = np.unique(df['subject'])
    n_k = len(st_remap)
    grp_heat = np.zeros((len(anis),len(st_remap),len(dists),2,len(conds),len(plt_types)))
    for p,(plt_type,plt_lim) in enumerate(zip(plt_types,plt_lims)):
        fig, axs = plt.subplots(3,len(conds),figsize=(10*len(conds),15))
        axs = axs.ravel()
        for c,cond in enumerate(conds): #cycle through conditions
            
            heatmap = np.zeros((len(anis),len(st_remap),len(dists)))
            scores = np.zeros((len(anis),len(dists)))
            coefs = np.zeros((len(anis),len(st_remap),len(dists)))
            suc_tr = np.zeros(len(dists))
            fail_tr = np.zeros(len(dists))
            for a,ani in enumerate(anis):
                for d,dist in enumerate(dists): #cycle through distances
                    temp_df = remove_aborts(df)
                    temp_df = temp_df[(temp_df['distance_DLC']==dist)&(temp_df[condition]==cond)&(temp_df['subject']==ani)]
                    temp_df.reset_index(inplace=True,drop=True)
                    y = temp_df['success'].to_numpy() #y = success
                    if len(y)>0:
                        x = np.zeros((len(temp_df),len(st_remap))) #x = trials by hmm freq at this distance
                        for idx,row in temp_df.iterrows():
                            for h,hmm_k in enumerate(st_remap):
                                x[idx,h] = row['hmm_%d_%s' % (hmm_k,plt_type)]

                        suc = x[y==1,:]
                        fail = x[y==0,:]
                        rel_freq = (np.nanmean(suc,axis=0)-np.nanmean(fail,axis=0))/(np.nanmean(suc,axis=0))
                        heatmap[a,:,d] = rel_freq
                        
                        grp_heat[a,:,d,0,c,p] = np.nanmean(fail,axis=0)
                        grp_heat[a,:,d,1,c,p] = np.nanmean(suc,axis=0)

                        suc_tr[d] = suc_tr[d] + len(suc)
                        fail_tr[d] = fail_tr[d] + len(fail)
                        
                        x[np.isnan(x)] = 0
                        reg = LinearRegression().fit(x, y)
                        scores[a,d] = reg.score(x,y)
                        coefs[a,:,d] = np.array(reg.coef_)
                    else:
                        heatmap[a,:,d] = np.zeros(len(st_remap))
                        scores[a,d] = 0
                        coefs[a,:,d] = np.zeros(len(st_remap))

            heatmap = np.nanmean(heatmap,axis=0)
            scores = np.nanmean(scores,axis=0)
            coefs = np.nanmean(coefs,axis=0)

            ax = axs[c]
            im = ax.imshow(heatmap,cmap='bwr',vmin=-plt_lim,vmax=plt_lim)
            ax.set_xticks(np.arange(0,len(dists)))
            ax.set_xticklabels(dists)
            ax.set_yticks(np.arange(0,len(st_remap)))
            ax.set_yticklabels(np.arange(1,len(st_remap)+1))
            ax.set_ylabel('hmm state')
            ax.set_xlabel('distance_DLC')
            ax.set_title(cond)
            fig.colorbar(im,ax=ax,label='%s (suc-fail)/suc' % (plt_type))
            
            ax = axs[c + len(conds)]
            ax.bar(np.arange(0,len(dists)),scores)
            ax.set_ylim(0,1)
            ax.set_yticks(np.arange(0,1.25,0.25))
            ax.set_xticks(np.arange(0,len(dists),1))
            ax.set_xticklabels(dists)
            ax.set_xlabel('gap distance (cm)')
            ax.set_ylabel('linear R2')
            ax.set_title('%s %s' % (cond,plt_type))
            for d in range(len(dists)):
                ax.text(d-0.4,0.4,'s=%d f=%d' % (suc_tr[d],fail_tr[d]),fontsize=12)
            
            ax = axs[c + 2*len(conds)]
            im = ax.imshow(coefs,cmap='bwr',vmin=-0.5,vmax=0.5)
            ax.set_xticks(np.arange(0,len(dists)))
            ax.set_xticklabels(dists)
            ax.set_yticks(np.arange(0,len(st_remap)))
            ax.set_yticklabels(np.arange(1,len(st_remap)+1))
            ax.set_ylabel('hmm state')
            ax.set_xlabel('distance_DLC')
            ax.set_title('%s %s' % (cond,plt_type))
            fig.colorbar(im,ax=ax,label='regression coefs')
            
        fig.suptitle('state %s difference success vs. failure %ds data' % (plt_type,time_per))

        if save_pdf:
            pp.savefig(fig)
            plt.close(fig)
    
    return fig, ax, grp_heat


def label_bob_frequency_multi(df,labels,cluster_key,cluster_nums):
    #create a column in labels corresponding to the frequency of a set of movements
    bob_freq = np.zeros((df.shape[0],)).astype('int')
    df['bob_freq'] = bob_freq
    for idx,row in df.iterrows():
        movs = labels[(labels['subject']==row['subject'])&(labels['expdate']==row['expdate'])& \
        (labels['trial']==row['trial'])][cluster_key].to_numpy()
        tot_n = 0
        for i in cluster_nums:
            n = len(np.where(movs==i)[0])
            tot_n += n
        tot_n
        df.loc[idx,'bob_freq']=tot_n

    return df

def plot_combined_mvmnts(labels,cluster_key,cluster_nums,plt_lim,save_pdf,pp):
    #plot a set of movement manually specified
    fig, axs = plt.subplots(1,len(cluster_nums),figsize=(5*len(cluster_nums),5))
    if len(cluster_nums)>1:
        axs = axs.ravel()
    for i,cn in enumerate(cluster_nums):
        
        tr = np.where(labels[cluster_key]==cn)[0]
        xtr = np.mean(labels['Side_x_eye_mvmnt'].iloc[tr])
        ytr = np.mean(labels['Side_y_eye_mvmnt'].iloc[tr])
        xtr = xtr - np.mean(xtr)
        ytr = ytr - np.mean(ytr)
    
        try:
            ax = axs[i]
        except:
            ax = axs
        ax.plot(xtr,ytr,'k')
        ax.plot(xtr[0],ytr[0],'bo',label='start')
        ax.plot(xtr[-1],ytr[-1],'ro',label='end')
        ax.axis([-plt_lim,plt_lim,-plt_lim,plt_lim])
        ax.set_xticks([-plt_lim,0,plt_lim])
        ax.set_yticks([-plt_lim,0,plt_lim])
        ax.set_xlabel('cm')
        ax.set_ylabel('cm')
        ax.set_title('mvmnt %d' % cn)
    ax.legend(fontsize=10)
    fig.tight_layout()

    if save_pdf:
        pp.savefig(fig)
        plt.close(fig)

    return fig, axs


def plot_fraction_success(df,condition,cols,ls,save_pdf,pp):
    anis = np.unique(df['subject'])
    dists = np.unique(df['distance_DLC'])
    conds = np.unique(df[condition])
    sucs = np.unique(df['success'])
    suc_lab = ['failure','success','abort']

    fig, axs = plt.subplots(1,len(sucs),figsize=(1.5*len(sucs),1.5))
    axs = axs.ravel()

    array = np.empty((len(anis),len(dists),len(conds),len(sucs)))
    array[:] = np.nan
    for a,ani in enumerate(anis):
        for d,dist in enumerate(dists):
            for c,cond in enumerate(conds):
                temp_df = df[(df['subject']==ani)&(df['distance_DLC']==dist)&(df[condition]==cond)]
                for s,suc in enumerate(sucs):
                    n_sucs = np.sum(temp_df['success']==suc)
                    sucs_fraction = n_sucs/temp_df.shape[0]
                    array[a,d,c,s] = sucs_fraction

    # print('alpha p=%0.3f' % (0.05/len(dists)))
    for s,suc in enumerate(sucs):
        ax = axs[s]
        for c,cond in enumerate(conds):
            ax.errorbar(x=dists,y=np.nanmean(array[:,:,c,s],axis=0),yerr=np.nanstd(array[:,:,c,s],axis=0)/np.sqrt(len(anis)),
                label=cond,color=cols[c],ls=ls,linewidth=0.75)
        for d,dist in enumerate(dists):
            st,p = stats.ttest_ind(array[:,d,0,s],array[:,d,1,s])
            # print('%s %dcm p=%0.3f'% (suc_lab[s],dist,p))

        ax.set_xlim(5,25)
        ax.set_ylim(0,1)
        ax.set_xlabel('gap distance (cm)')
        ax.set_ylabel('fraction of trials')
        # ax.set_title('%s' % suc_lab[s])
        # ax.legend(fontsize=8)
        ax = xy_axis(ax)

    # plt.suptitle(suptitle)
    fig.tight_layout()
    if save_pdf:
        pp.savefig(fig)
        plt.close(fig)

    return fig, axs, array

def anova_fraction_success(df_cond,array,condition):
    anis = np.unique(df_cond['subject'])
    dists = np.unique(df_cond['distance_DLC'])
    conds = np.unique(df_cond[condition])
    sucs = np.unique(df_cond['success'])
    suc_lab = ['failure','success','abort']

    anova_suc = np.zeros(len(sucs))
    anova_F = np.zeros(len(sucs))
    for s,suc in enumerate(sucs):
        anova_df = pd.DataFrame(columns=['distance_DLC',condition,'success'])
        for d,dist in enumerate(dists):
            for c,cond in enumerate(conds):
                temp_df = pd.DataFrame(columns=['distance_DLC',condition,'success'])
                temp_df['distance_DLC']  = pd.Series(np.repeat(dist,len(anis)))
                temp_df[condition] = pd.Series(np.repeat(cond,len(anis)))
                temp_df['success'] = array[:,d,c,s]
                anova_df = pd.concat([anova_df,temp_df],axis=0)
        #print anova results
        model = ols('success ~ C(distance_DLC) + C(%s) + C(distance_DLC):C(%s)' % (condition,condition), data=anova_df).fit()
        anova_suc[s] = sm.stats.anova_lm(model, typ=2).loc['C(%s)' % condition,'PR(>F)']
        anova_F[s] = sm.stats.anova_lm(model, typ=2).loc['C(%s)' % condition,'F']
        # print(suc_lab[s])
        # print(sm.stats.anova_lm(model, typ=2))
        # print('')
    return anova_suc, anova_F

def plot_jump_accuracy(og_df,condition,manipulation,plot_params,save_pdf,pp,suptitle):
    ### Calculate and plot the jumping distance error given movement cluster k_clust at each distance

    ### INPUTS
    ### df: the dataframe with all of the data
    ### condition: the key from df that you want to split the data into (e.g. 'ocular')
    ### manipulation: e.g. laser, success etc.
    ### save_pdf: whether option was selected to save pdf
    ### pp: the pdf object (only necessary if saving PDF)

    ### OUTPUTS
    ### fig, ax: the figure and axes objects
    df = remove_aborts(og_df)
    anis = np.unique(df['subject'])
    dists = np.unique(df['distance_DLC'])
    conds = np.unique(df[condition])
    mans = np.unique(df[manipulation])

    fig, axs = plt.subplots(1,len(conds),figsize=(len(conds)*5,5))
    axs = axs.ravel()

    for c,cond in enumerate(conds):
        ax = axs[c]
        stats_array = np.empty((len(anis),len(dists),len(mans)))
        stats_array[:] = np.nan
        anova_df = pd.DataFrame(columns=['distance_DLC',manipulation,'accuracy'])
 
        for m,man in enumerate(mans):
            jumpcurve = df[(df[manipulation]==man)&(df[condition]==cond)].groupby(['subject','distance_DLC']).mean()
            # print(jumpcurve)
            # jumpcurve.reset_index(inplace=True)
            mnplot = jumpcurve.groupby(['distance_DLC']).mean()
            semplot = jumpcurve.groupby(['distance_DLC']).std()/np.sqrt(len(anis))
            mnplot.reset_index(inplace=True)
            semplot.reset_index(inplace=True)

            ax.errorbar(mnplot['distance_DLC'],
                mnplot['accuracy'],
                yerr=semplot['accuracy'],
                label=manipulation + ' %s' % man,
                color=plot_params['cond_col'][m],linewidth=1,zorder=c) # this only works for manipulation having two types...

            # for d,dist in enumerate(dists):
            #     stats_array[:,d,m] = jumpcurve.xs(dist)['accuracy']

            #     temp_df = pd.DataFrame(columns=['distance_DLC',manipulation,'accuracy'])
            #     temp_df['distance_DLC']  = pd.Series(np.repeat(dist,len(anis)))
            #     temp_df[manipulation] = pd.Series(np.repeat(man,len(anis)))
            #     temp_df['accuracy'] = pd.Series(jumpcurve.xs(dist)['accuracy'])#.to_numpy())
            #     anova_df = pd.concat([anova_df,temp_df],axis=0)
        ax.set_xlabel('gap distance (cm)')
        ax.set_ylabel('error (cm)')
        ax.set_title(('%s ' +  manipulation) % cond)
        ax.set_xlim(6,29)
        ax.set_ylim(-5,10)
        ax.legend(fontsize=10)
        plt.xticks(np.arange(10, 30, step=5))
        #         plt.yticks(np.arange(0, 1.5, step=0.5))
        ax = xy_axis(ax)

        # print(jumpcurve,anova_df)
        # #print anova results
        # model = ols('accuracy ~ C(distance) + C(manipulation) + C(distance):C(manipulation)', data=anova_df).fit()
        # print(mans)
        # print(sm.stats.anova_lm(model, typ=2))
        # print('')

        # for d,dist in enumerate(dists):
        #     s, p = stats.ttest_rel(stats_array[:,d,0],stats_array[:,d,1])
        #     print('distance %d, p=%0.3f' % (dist,p))
        # print('alpha=%0.3f' % (0.05/len(dists)))

    fig.suptitle(suptitle)
    fig.tight_layout()

    

    if save_pdf:
        pp.savefig(fig)
        plt.close(fig)

    return fig, ax


def plot_performance_manipulation(axs,df,condition,manipulation,aborts,plt_min,plt_max,color_scheme):#,save_pdf,pp,suptitle):
    ### Calculate and plot the jumping distance error given movement cluster k_clust at each distance

    ### INPUTS
    ### df: the dataframe with all of the data
    ### condition: the key from df that you want to split the data into (e.g. 'ocular')
    ### manipulation: e.g. laser, success etc.
    ### save_pdf: whether option was selected to save pdf
    ### pp: the pdf object (only necessary if saving PDF)

    ### OUTPUTS
    ### fig, ax: the figure and axes objects

    anis = np.unique(df['subject'])
    dists = np.unique(df['distance_DLC'])
    conds = np.unique(df[condition])

    # fig, axs = plt.subplots(1,len(conds),figsize=(len(conds)*5,5))
    # axs = axs.ravel()

    if aborts:
        temp_df = aborts_as_failures(df)
    else:
        temp_df = remove_aborts(df)

    for c,cond in enumerate(conds):
        ax = axs[c]
        for m,man in enumerate(np.unique(df[manipulation])):
            mn = temp_df[(df[manipulation]==man)&(df[condition]==cond)].groupby(['distance_DLC','subject']).mean()
            mnplot = mn.groupby(['distance_DLC']).mean()#apply(lambda x: np.mean(x))
            mnplot.reset_index(inplace=True)
            semplot = mn.groupby(['distance_DLC']).std()/np.sqrt(len(anis))#apply(lambda x: np.std(x)/np.sqrt(len(anis)))
            semplot.reset_index(inplace=True)
            try:
                ax.errorbar(mnplot['distance_DLC'],
                            mnplot['success'],
                            yerr=semplot['success'],
                            label=manipulation + ' %s' % man,
                            color=color_scheme[m],linewidth=1,zorder=c) # this only works for manipulation having two types...
            except:
                pass
            mn.reset_index(inplace=True)
            for ani in anis:
                ax.plot(mn[mn['subject']==ani]['distance_DLC'],mn[mn['subject']==ani]['success'],'-',color=color_scheme[m],linewidth=0.25,alpha=0.5)
        ax.set_xlabel('gap distance (cm)')
        ax.set_ylabel('success rate')
        # ax.set_title(('%s ' +  manipulation) % cond)
        ax.set_xlim(plt_min,plt_max)
        ax.set_ylim(0,1.1)
        # ax.legend(fontsize=10)
        ax.set_xticks(np.arange(10,30,5))
        #         plt.yticks(np.arange(0, 1.5, step=0.5))
        ax = xy_axis(ax)

    # fig.suptitle(suptitle)

    # plt.tight_layout()

    # if save_pdf:
    #     pp.savefig(fig)
    #     plt.close(fig)

    return axs#fig, ax




def plot_jumpdist_manipulation(df,condition,manipulation,color_scheme,save_pdf,pp,suptitle):
    ### Calculate and plot the jumping distance error given movement cluster k_clust at each distance

    ### INPUTS
    ### df: the dataframe with all of the data
    ### condition: the key from df that you want to split the data into (e.g. 'ocular')
    ### manipulation: e.g. laser, success etc.
    ### save_pdf: whether option was selected to save pdf
    ### pp: the pdf object (only necessary if saving PDF)

    ### OUTPUTS
    ### fig, ax: the figure and axes objects

    anis = np.unique(df['subject'])
    dists = np.unique(df['distance_DLC'])
    conds = np.unique(df[condition])

    fig, axs = plt.subplots(1,len(conds),figsize=(len(conds)*5,5))
    axs = axs.ravel()

    temp_df = remove_aborts(df)
    for c,cond in enumerate(conds):
        ax = axs[c]
        ax.plot([6,29],[6,29],':',color=[0.5,0.5,0.5])
        for man in np.unique(df[manipulation]):
            mn = temp_df[(df[manipulation]==man)&(df[condition]==cond)].groupby(['distance_DLC']).mean()
            mnplot = mn.groupby(['distance_DLC']).mean()
            mnplot.reset_index(inplace=True)
            semplot = mn.groupby(['distance_DLC']).std()/np.sqrt(len(anis))
            semplot.reset_index(inplace=True)

            try:
                ax.errorbar(mnplot['distance_DLC'],
                            mnplot['jumpdist'],
                            yerr=semplot['jumpdist'],
                            label=manipulation + ' %s' % man,
                            color=color_scheme[man],linewidth=1,zorder=c) # this only works for manipulation having two types...
            except:
                pass
            mn.reset_index(inplace=True)
            for ani in anis:
                ax.plot(mn[mn['subject']==ani]['distance_DLC'],mn[mn['subject']==ani]['jumpdist'],'-',color=color_scheme[m],linewidth=0.25,alpha=0.5)
        ax.set_xlabel('gap distance (cm)')
        ax.set_ylabel('distance jumped (cm)')
        # ax.set_title(('%s ' +  manipulation) % cond)
        ax.set_xlim(6,29)
        ax.set_ylim(6,29)
        # ax.legend(fontsize=10)
        plt.xticks(np.arange(10, 30, step=5))
        plt.yticks(np.arange(10, 30, step=5))
        ax = xy_axis(ax)

    fig.suptitle(suptitle)

    # plt.tight_layout()

    if save_pdf:
        pp.savefig(fig)
        plt.close(fig)

    return fig, ax



def dlc_file_from_row(vid_dir,side,row):
    try:
        fname = find(row['expdate'] + '_' + row['subject'] + '_' + row['condition'] + '_' + str(row['trial']-1) + '_' + side + '*filtered.h5',vid_dir)[0] #trial is 0 indexed in file, 1 in df
    except:
        fname = find(row['expdate'] + '_' + row['subject'] + '_' + row['condition'] + '_' + side + '_' + str(row['trial']-1) + 'DLC*filtered.h5',vid_dir)[0]
    return fname

###old version
def dlc_file_from_row_og(vid_dir,side,row):
    fname = os.path.join(vid_dir, row['expdate'] + '_' + row['subject'] + '_' + side + '_' + str(row['trial']).zfill(3) + 'DLC_resnet50_Jumping2Jun30shuffle1_1030000_filtered.h5')
    return fname



# plot all jumps

# def vidname_from_row(vid_dir,side,row):
#     vidname = os.path.join(vid_dir,
#                            row['expdate'] + '_' + row['subject'] + '_' + side + '_' + str(row['trial']).zfill(3) + '.avi')
#     return vidname

def vidname_from_row(vid_dir,side,row):
    try:
        fname = find(row['expdate'] + '_' + row['subject'] + '_' + row['condition'] + '_' + str(row['trial']-1) + '_' + side + '*.avi',vid_dir)[0] #trial is 0 indexed in file, 1 in df
    except:
        fname = find(row['expdate'] + '_' + row['subject'] + '_' + row['condition'] + '_' + side + '_' + str(row['trial']-1) + '*.avi',vid_dir)[0]
    return fname


def grab_vidframe(file,frame_num):
    
    ### grab a video frame to overlay example points
    vid = cv2.VideoCapture(file)
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    fps = int(vid.get(cv2.CAP_PROP_FPS))

    vid.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = vid.read()
    vid.release()

    return frame,fps,frame_width,frame_height,ret


def plot_landing_position_change_mean(axs,df_cond,manipulation,x_min,x_max,y_min,y_max,color_scheme):
    axs = axs.ravel()

    dists = np.unique(df_cond['distance_DLC'])
    anis = np.unique(df_cond['subject'])
    mans = np.unique(df_cond[manipulation])

    mn_land_xy_diff = np.zeros((len(dists),2,int(len(anis)/2)))
    mn_land_x_diff = np.zeros((len(dists),2,int(len(anis)/2)))
    mn_land_y_diff = np.zeros((len(dists),2,int(len(anis)/2)))
    std_land_xy_diff = np.zeros((len(dists),2,int(len(anis)/2)))
    std_land_x = np.zeros((len(dists),2,2,int(len(anis)/2))) #dists,man,group,animal
    std_land_y = np.zeros((len(dists),2,2,int(len(anis)/2)))

    wt_count = -1
    ch_count = -1

    for a,ani in enumerate(anis):
        df_ani = df_cond[(df_cond['subject']==ani)&(df_cond['success']==1)]
        df_ani.reset_index(inplace=True,drop=True)

        if df_ani['expt_group'].iloc[0]=='WT':
            grp='WT'
            wt_count += 1
        else:
            grp='ChR2'
            ch_count += 1

        for d,dist in enumerate(dists):
            mn_land = np.zeros(4) #x1,x2,y1,y2
            std_land = np.zeros(4) #x1,x2,y1,y2
            for m,man in enumerate(mans):
                temp_df = df_ani[(df_ani['distance_DLC']==dist)&(df_ani[manipulation]==man)]
                temp_df.reset_index(inplace=True,drop=True)

                #get landing positions
                xs = [i[-1] for i in temp_df['Top LEar x']]
                ys = [i[-1] for i in temp_df['Top LEar y']]
        
                mn_land[m] = np.nanmean(xs)/df_ani['Top_pixpercm'].iloc[0]
                mn_land[m+2] = np.nanmean(ys)/df_ani['Top_pixpercm'].iloc[0]
                std_land[m] = np.nanstd(xs)/df_ani['Top_pixpercm'].iloc[0]
                std_land[m+2] = np.nanstd(ys)/df_ani['Top_pixpercm'].iloc[0]
            if grp=='WT':
                mn_land_xy_diff[d,0,wt_count] = np.sqrt((mn_land[1]-mn_land[0])**2 + (mn_land[3]-mn_land[2])**2)
                mn_land_x_diff[d,0,wt_count] = np.abs(mn_land[1]-mn_land[0])
                mn_land_y_diff[d,0,wt_count] = np.abs(mn_land[3]-mn_land[2])
                # std_land_xy_diff[d,0,wt_count] = np.sqrt((std_land[1]-std_land[0])**2 + (std_land[3]-std_land[2])**2)
                std_land_xy_diff[d,0,wt_count] = np.abs((std_land[1]+std_land[3])-(std_land[0]+std_land[2]))
                std_land_x[d,0,0,wt_count] = std_land[0]
                std_land_x[d,1,0,wt_count] = std_land[1]
                std_land_y[d,0,0,wt_count] = std_land[2]
                std_land_y[d,1,0,wt_count] = std_land[3]
            else:
                mn_land_xy_diff[d,1,ch_count] = np.sqrt((mn_land[1]-mn_land[0])**2 + (mn_land[3]-mn_land[2])**2)
                mn_land_x_diff[d,1,ch_count] = np.abs(mn_land[1]-mn_land[0])
                mn_land_y_diff[d,1,ch_count] = np.abs(mn_land[3]-mn_land[2])
                # std_land_xy_diff[d,1,ch_count] = np.sqrt((std_land[1]-std_land[0])**2 + (std_land[3]-std_land[2])**2)
                std_land_xy_diff[d,1,ch_count] = np.abs((std_land[1]+std_land[3])-(std_land[0]+std_land[2]))
                std_land_x[d,0,1,ch_count] = std_land[0]
                std_land_x[d,1,1,ch_count] = std_land[1]
                std_land_y[d,0,1,ch_count] = std_land[2]
                std_land_y[d,1,1,ch_count] = std_land[3]

    ax = axs[0]
    ax.errorbar(x=dists,y=np.nanmean(mn_land_xy_diff[:,0,:],axis=1),yerr=np.nanstd(mn_land_xy_diff[:,0,:],axis=1)/np.sqrt(wt_count+1),color=color_scheme[0],linewidth=1,label='WT',zorder=0)
    ax.errorbar(x=dists,y=np.nanmean(mn_land_xy_diff[:,1,:],axis=1),yerr=np.nanstd(mn_land_xy_diff[:,1,:],axis=1)/np.sqrt(ch_count+1),color=color_scheme[1],linewidth=1,label='ChR2',zorder=1)
    for a in range(mn_land_xy_diff.shape[-1]):
        ax.plot(dists,mn_land_xy_diff[:,0,a],color=color_scheme[0],linewidth=0.25,alpha=0.5)
        ax.plot(dists,mn_land_xy_diff[:,1,a],color=color_scheme[1],linewidth=0.25,alpha=0.5)
    ax.set_xlim(x_min,x_max)
    ax.set_xticks(np.arange(10,30,5))
    ax.set_ylim(y_min,y_max)
    ax.set_xlabel('gap distance')
    ax.set_ylabel('dmean land (cm)')
    # ax.legend()
    ax = xy_axis(ax)

    ax = axs[1]
    ax.errorbar(x=dists,y=np.nanmean(std_land_xy_diff[:,0,:],axis=1),yerr=np.nanstd(std_land_xy_diff[:,0,:],axis=1)/np.sqrt(wt_count+1),color=color_scheme[0],linewidth=1,label='WT',zorder=0)
    ax.errorbar(x=dists,y=np.nanmean(std_land_xy_diff[:,1,:],axis=1),yerr=np.nanstd(std_land_xy_diff[:,1,:],axis=1)/np.sqrt(ch_count+1),color=color_scheme[1],linewidth=1,label='ChR2',zorder=1)
    for a in range(mn_land_xy_diff.shape[-1]):
        ax.plot(dists,mn_land_xy_diff[:,0,a],color=color_scheme[0],linewidth=0.25,alpha=0.5)
        ax.plot(dists,mn_land_xy_diff[:,1,a],color=color_scheme[1],linewidth=0.25,alpha=0.5)
    ax.set_xlim(x_min,x_max)
    ax.set_xticks(np.arange(10,30,5))
    ax.set_ylim(y_min,y_max)
    ax.set_xlabel('gap distance')
    ax.set_ylabel('std land (cm)')
    # ax.legend()
    ax = xy_axis(ax)

    # s,p = stats.ttest_rel(np.nanmean(mn_land_xy_diff[:,0,:],axis=0),np.nanmean(mn_land_xy_diff[:,1,:],axis=0))
    # print('laser off vs. on mean position p=%0.3f' % p)

    s,p = stats.ttest_rel(np.nanmean(std_land_xy_diff[:,0,:],axis=0),np.nanmean(std_land_xy_diff[:,1,:],axis=0))
    print('laser off vs. on std position p=%0.3f' % p)

    return axs, mn_land_x_diff, mn_land_y_diff, mn_land_xy_diff, std_land_x, std_land_y, std_land_xy_diff


def plot_landing_position_change_mean_og(axs,df_cond,manipulation,x_min,x_max,y_min,y_max,color_scheme):
    axs = axs.ravel()

    dists = np.unique(df_cond['distance_DLC'])
    anis = np.unique(df_cond['subject'])
    mans = np.unique(df_cond[manipulation])

    mn_land_xy_diff = np.zeros((len(dists),2,int(len(anis)/2)))
    mn_land_x_diff = np.zeros((len(dists),2,int(len(anis)/2)))
    mn_land_y_diff = np.zeros((len(dists),2,int(len(anis)/2)))
    std_land_xy_diff = np.zeros((len(dists),2,int(len(anis)/2)))
    std_land_x = np.zeros((len(dists),2,2,int(len(anis)/2))) #dists,man,group,animal
    std_land_y = np.zeros((len(dists),2,2,int(len(anis)/2)))

    wt_count = -1
    ch_count = -1

    for a,ani in enumerate(anis):
        df_ani = df_cond[(df_cond['subject']==ani)&(df_cond['success']==1)]
        df_ani.reset_index(inplace=True,drop=True)

        if df_ani['expt_group'].iloc[0]=='WT':
            grp='WT'
            wt_count += 1
        else:
            grp='ChR2'
            ch_count += 1

        for d,dist in enumerate(dists):
            mn_land = np.zeros(4) #x1,x2,y1,y2
            std_land = np.zeros(4) #x1,x2,y1,y2
            for m,man in enumerate(mans):
                temp_df = df_ani[(df_ani['distance_DLC']==dist)&(df_ani[manipulation]==man)]
                temp_df.reset_index(inplace=True,drop=True)

                #get landing positions
                xs = [i[int(e)-int(s)] for i,e,s in zip(temp_df['Top LEar x'],temp_df['Top_End'],temp_df['Top_Start'])]
                ys = [i[int(e)-int(s)] for i,e,s in zip(temp_df['Top LEar y'],temp_df['Top_End'],temp_df['Top_Start'])]
        
                mn_land[m] = np.nanmean(xs)/df_ani['Top_pixpercm'].iloc[0]
                mn_land[m+2] = np.nanmean(ys)/df_ani['Top_pixpercm'].iloc[0]
                std_land[m] = np.nanstd(xs)/df_ani['Top_pixpercm'].iloc[0]
                std_land[m+2] = np.nanstd(ys)/df_ani['Top_pixpercm'].iloc[0]
            if grp=='WT':
                mn_land_xy_diff[d,0,wt_count] = np.sqrt((mn_land[1]-mn_land[0])**2 + (mn_land[3]-mn_land[2])**2)
                mn_land_x_diff[d,0,wt_count] = np.abs(mn_land[1]-mn_land[0])
                mn_land_y_diff[d,0,wt_count] = np.abs(mn_land[3]-mn_land[2])
                # std_land_xy_diff[d,0,wt_count] = np.sqrt((std_land[1]-std_land[0])**2 + (std_land[3]-std_land[2])**2)
                std_land_xy_diff[d,0,wt_count] = np.abs((std_land[1]+std_land[3])-(std_land[0]+std_land[2]))
                std_land_x[d,0,0,wt_count] = std_land[0]
                std_land_x[d,1,0,wt_count] = std_land[1]
                std_land_y[d,0,0,wt_count] = std_land[2]
                std_land_y[d,1,0,wt_count] = std_land[3]
            else:
                mn_land_xy_diff[d,1,ch_count] = np.sqrt((mn_land[1]-mn_land[0])**2 + (mn_land[3]-mn_land[2])**2)
                mn_land_x_diff[d,1,ch_count] = np.abs(mn_land[1]-mn_land[0])
                mn_land_y_diff[d,1,ch_count] = np.abs(mn_land[3]-mn_land[2])
                # std_land_xy_diff[d,1,ch_count] = np.sqrt((std_land[1]-std_land[0])**2 + (std_land[3]-std_land[2])**2)
                std_land_xy_diff[d,1,ch_count] = np.abs((std_land[1]+std_land[3])-(std_land[0]+std_land[2]))
                std_land_x[d,0,1,ch_count] = std_land[0]
                std_land_x[d,1,1,ch_count] = std_land[1]
                std_land_y[d,0,1,ch_count] = std_land[2]
                std_land_y[d,1,1,ch_count] = std_land[3]

    ax = axs[0]
    ax.errorbar(x=dists,y=np.nanmean(mn_land_xy_diff[:,0,:],axis=1),yerr=np.nanstd(mn_land_xy_diff[:,0,:],axis=1)/np.sqrt(wt_count+1),color=color_scheme[0],linewidth=1,label='WT',zorder=0)
    ax.errorbar(x=dists,y=np.nanmean(mn_land_xy_diff[:,1,:],axis=1),yerr=np.nanstd(mn_land_xy_diff[:,1,:],axis=1)/np.sqrt(ch_count+1),color=color_scheme[1],linewidth=1,label='ChR2',zorder=1)
    for a in range(mn_land_xy_diff.shape[-1]):
        ax.plot(dists,mn_land_xy_diff[:,0,a],color=color_scheme[0],linewidth=0.25,alpha=0.5)
        ax.plot(dists,mn_land_xy_diff[:,1,a],color=color_scheme[1],linewidth=0.25,alpha=0.5)
    ax.set_xlim(x_min,x_max)
    ax.set_xticks(np.arange(10,30,5))
    ax.set_ylim(y_min,y_max)
    ax.set_xlabel('gap distance')
    ax.set_ylabel('dmean land (cm)')
    # ax.legend()
    ax = xy_axis(ax)

    ax = axs[1]
    ax.errorbar(x=dists,y=np.nanmean(std_land_xy_diff[:,0,:],axis=1),yerr=np.nanstd(std_land_xy_diff[:,0,:],axis=1)/np.sqrt(wt_count+1),color=color_scheme[0],linewidth=1,label='WT',zorder=0)
    ax.errorbar(x=dists,y=np.nanmean(std_land_xy_diff[:,1,:],axis=1),yerr=np.nanstd(std_land_xy_diff[:,1,:],axis=1)/np.sqrt(ch_count+1),color=color_scheme[1],linewidth=1,label='ChR2',zorder=1)
    for a in range(mn_land_xy_diff.shape[-1]):
        ax.plot(dists,mn_land_xy_diff[:,0,a],color=color_scheme[0],linewidth=0.25,alpha=0.5)
        ax.plot(dists,mn_land_xy_diff[:,1,a],color=color_scheme[1],linewidth=0.25,alpha=0.5)
    ax.set_xlim(x_min,x_max)
    ax.set_xticks(np.arange(10,30,5))
    ax.set_ylim(y_min,y_max)
    ax.set_xlabel('gap distance')
    ax.set_ylabel('std land (cm)')
    # ax.legend()
    ax = xy_axis(ax)

    s,p = stats.ttest_rel(np.nanmean(std_land_xy_diff[:,0,:],axis=0),np.nanmean(std_land_xy_diff[:,1,:],axis=0))
    print('laser off vs. on std position p=%0.3f' % p)

    return axs, mn_land_x_diff, mn_land_y_diff, mn_land_xy_diff, std_land_x, std_land_y, std_land_xy_diff


def plot_jumps(df,vid_dir,side,save_pdf,pp,suptitle=''):
    suc_col = [[0.5,0.5,0.5],'k']
    dists = np.unique(df['distance_DLC'])
    ocs = np.unique(df['ocular'])
    for ani in np.unique(df['subject']):
        fig, axs = plt.subplots(len(ocs),len(dists),figsize=(3*len(dists),3*len(ocs)))
        for o,oc in enumerate(ocs):
            for d,dist in enumerate(dists):
                temp_df = df[(df['subject']==ani)&(df['ocular']==oc)&(df['distance_DLC']==dist)]
                temp_df.reset_index(inplace=True)
                temp_df = remove_aborts(temp_df)

                vidcnt = np.ones((len(ocs),len(dists)))
                for index,row in temp_df.iterrows():
                    start = row[side + '_Start']
                    jump = row[side + '_Jump']
                    end = row[side + '_End']

                    ax = axs[o,d]
                    if (vidcnt[o,d]==1)&(row['platform_DLC']==2):
                        vidfile = vidname_from_row(vid_dir,side,row)
                        frame,fps,frame_width,frame_height,ret = grab_vidframe(vidfile,int(jump-start))
                        ax.imshow(frame)
                        ax.axis([0,1440,1080-250,150])
                        vidcnt[o,d]=0

                    xtr = row[side + ' LEye x'].copy()
                    ytr = row[side + ' LEye y'].copy()

                    ax.plot(xtr,ytr,'r',linewidth=0.5,alpha=0.5)
                    ax.plot(xtr[-1],ytr[-1],'b.')

        [axi.set_axis_off() for axi in axs.ravel()]
        fig.suptitle(ani + ' ' + oc + ' ' + side + ' ' + suptitle)
        fig.tight_layout()
        if save_pdf:
            pp.savefig(fig,dpi=300)
            plt.close(fig)

    
# def plot_jumps(df,side,vid_path,condition,plot_params,save_pdf,pp):
#     conds = np.unique(df[condition])
#     cols = ['k','b','g','m','r']
#     vid = find(df['expdate'][0] + '_' + df['subject'][0] + '_' + side + '*.avi',vid_path)[0]
#     frame,fps,frame_width,frame_height = grab_vidframe(vid,10)
#     for d,dist in enumerate(np.unique(df['distance'])):
#         fig, axs = plt.subplots(1,2,figsize=(2*frame_width/100,frame_height/100))
#         axs = axs.ravel()
#         for suc in range(2):
#             ax = axs[suc]
#             temp_df = df[(df['distance']==dist)&(df['success']==suc)]
#             temp_df.reset_index(inplace=True,drop=True)
#             try:
#                 vid = os.path.join(vid_path,temp_df['expdate'][0] + '_' + temp_df['subject'][0] + '_'+ side + '_' + str(temp_df['trial'][0]).zfill(3) + '.avi')
#                 frame,fps,frame_width,frame_height = grab_vidframe(vid,10)
#                 ax.imshow(frame)
#                 for c,cond in enumerate(conds):
#                     ax.plot(0,0,'.',color=plot_params['plat_cols'][c],label=cond)
#                     for x,y,start,jump,end in zip(temp_df[temp_df[condition]==cond][side + ' TailBase x'],
#                         temp_df[temp_df[condition]==cond][side + ' TailBase y'],
#                         temp_df[temp_df[condition]==cond][side + '_Start'],
#                         temp_df[temp_df[condition]==cond][side + '_Jump'],
#                         temp_df[temp_df[condition]==cond][side + '_End']):
#                         ax.plot(x[jump-start:end-start],y[jump-start:end-start],color=plot_params['plat_cols'][c])
#                 ax.set_title('success=%d' % suc)
#                 ax.legend()
#                 ax.axis('off')
#             except:
#                 pass

#         fig.suptitle('%s distance=%d cm' % (df['subject'][0],dist))

#         # plt.tight_layout()

#         if save_pdf:
#             pp.savefig(fig)
#             plt.close(fig)

#     return fig, ax

def smooth_trace(trace,like,lh_thresh):
    box_sz = 5 #for box filtering
    box = np.ones(box_sz)/box_sz
    trace[like<lh_thresh] = np.nan #nan out low likelihoods
    trace = pd.Series(trace).interpolate().to_numpy() #interp over nans
    trace = pd.Series(trace).fillna(method='bfill').to_numpy() #if first vals are nans fill them
    trace = signal.medfilt(trace,kernel_size=3) #mean filter
    trace = np.convolve(trace, box, mode='same') #smoothing filter

    return trace


def calculate_pitch(df_all,vid_dir):
    lh_thresh = 0.95
    start_fr = 2#
    side_pitch = []
    side_position = []

    for index,row in df_all.iterrows():
        if np.mod(index,500)==0:
            print('doing %d of %d' % (index,len(df_all)))

        eye_x,eye_y,ear_x,ear_y = eye_ear_from_row(row,vid_dir,'Side',lh_thresh)

        # get head angle
        rho = np.degrees(np.arctan2(ear_y-eye_y,ear_x-eye_x))
        # rho[rho_like] = np.nan

        side_pitch.append(rho)
        side_position.append(eye_y)

    df_all['Side_pitch'] = side_pitch
    df_all['Side_pitch_mean'] = [np.nanmean(r[2:-2]) for r in side_pitch]
    df_all['Side_pitch_std'] = [np.nanstd(r[2:-2]) for r in side_pitch]

    return df_all


def calculate_pitch_og(df_all,vid_dir):
    lh_thresh = 0.95
    start_fr = 2#
    side_pitch = []
    side_position = []

    for index,row in df_all.iterrows():
        if np.mod(index,500)==0:
            print('doing %d of %d' % (index,len(df_all)))

        eye_x,eye_y,ear_x,ear_y = eye_ear_from_row_og(row,vid_dir,'Side',lh_thresh)

        # get head angle
        rho = np.degrees(np.arctan2(ear_y-eye_y,ear_x-eye_x))
        # rho[rho_like] = np.nan

        side_pitch.append(rho)
        side_position.append(eye_y)

    df_all['Side_pitch'] = side_pitch
    df_all['Side_pitch_mean'] = [np.nanmean(r[2:-2]) for r in side_pitch]
    df_all['Side_pitch_std'] = [np.nanstd(r[2:-2]) for r in side_pitch]

    return df_all


def calculate_yaw(df_all,vid_dir):
    lh_thresh = 0.95
    start_fr = 2#-200
    top_yaw = []

    for index,row in df_all.iterrows():
        if np.mod(index,500)==0:
            print('doing %d of %d' % (index,len(df_all)))

        jump_fr = int(row['Top_Jump']-row['Top_Start'])

        eye_x,eye_y,ear_x,ear_y = eye_ear_from_row(row,vid_dir,'Top',lh_thresh)

        # get head angle
        rho = np.degrees(np.arctan2(ear_y-eye_y,ear_x-eye_x))
        # rho[rho_like] = np.nan

        top_yaw.append(rho)

    df_all['Top_yaw'] = top_yaw
    df_all['Top_yaw_mn'] = [np.nanmean(r) for r in top_yaw]
    df_all['Top_yaw_std'] = [np.nanstd(r) for r in top_yaw]

    return df_all


def calculate_yaw_og(df_all,vid_dir):
    lh_thresh = 0.95
    start_fr = 2#-200
    top_yaw = []

    for index,row in df_all.iterrows():
        if np.mod(index,500)==0:
            print('doing %d of %d' % (index,len(df_all)))

        jump_fr = int(row['Top_Jump']-row['Top_Start'])

        eye_x,eye_y,ear_x,ear_y = eye_ear_from_row_og(row,vid_dir,'Top',lh_thresh)

        # get head angle
        rho = np.degrees(np.arctan2(ear_y-eye_y,ear_x-eye_x))
        # rho[rho_like] = np.nan

        top_yaw.append(rho)

    df_all['Top_yaw'] = top_yaw
    df_all['Top_yaw_mn'] = [np.nanmean(r) for r in top_yaw]
    df_all['Top_yaw_std'] = [np.nanstd(r) for r in top_yaw]

    return df_all



def eye_ear_from_row(row,vid_dir,side,lh_thresh):
    dlc_file = dlc_file_from_row(vid_dir,side,row)
    pts = pd.read_hdf(dlc_file)
    pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]
    eye_x = pts['LEye x'][int(row['%s_Start' % side]):int(row['%s_Jump' % side])].to_numpy()
    eye_y = pts['LEye y'][int(row['%s_Start' % side]):int(row['%s_Jump' % side])].to_numpy()
    eye_like = pts['LEye likelihood'][int(row['%s_Start' % side]):int(row['%s_Jump' % side])].to_numpy()
    ear_x = pts['LEar x'][int(row['%s_Start' % side]):int(row['%s_Jump' % side])].to_numpy()
    ear_y = pts['LEar y'][int(row['%s_Start' % side]):int(row['%s_Jump' % side])].to_numpy()
    ear_like =  pts['LEar likelihood'][int(row['%s_Start' % side]):int(row['%s_Jump' % side])].to_numpy()

    eye_x = smooth_trace(eye_x,eye_like,lh_thresh)
    eye_y = smooth_trace(eye_y,eye_like,lh_thresh)
    ear_x = smooth_trace(ear_x,ear_like,lh_thresh)
    ear_y = smooth_trace(ear_y,ear_like,lh_thresh)

    return eye_x,eye_y,ear_x,ear_y


def eye_ear_from_row_og(row,vid_dir,side,lh_thresh):
    dlc_file = dlc_file_from_row_og(vid_dir,side,row)
    pts = pd.read_hdf(dlc_file)
    pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]
    eye_x = pts['LEye x'][:int(row['%s_Jump' % side])-int(row['%s_Start' % side])].to_numpy()
    eye_y = pts['LEye y'][:int(row['%s_Jump' % side])-int(row['%s_Start' % side])].to_numpy()
    eye_like = pts['LEye likelihood'][:int(row['%s_Jump' % side])-int(row['%s_Start' % side])].to_numpy()
    ear_x = pts['LEar x'][:int(row['%s_Jump' % side])-int(row['%s_Start' % side])].to_numpy()
    ear_y = pts['LEar y'][:int(row['%s_Jump' % side])-int(row['%s_Start' % side])].to_numpy()
    ear_like =  pts['LEar likelihood'][:int(row['%s_Jump' % side])-int(row['%s_Start' % side])].to_numpy()

    eye_x = smooth_trace(eye_x,eye_like,lh_thresh)
    eye_y = smooth_trace(eye_y,eye_like,lh_thresh)
    ear_x = smooth_trace(ear_x,ear_like,lh_thresh)
    ear_y = smooth_trace(ear_y,ear_like,lh_thresh)

    return eye_x,eye_y,ear_x,ear_y

### movie making code

def make_path(base,row):
    if row['success']==1:
        suc = 'success'
    else:
        suc = 'fail'
    
    pathname = os.path.join(base,row['subject'] + '\\' + suc)
    
    if not os.path.exists(pathname):
        os.makedirs(pathname)
        
    return pathname
        
# Draw points on frame
def draw_points(frame, x, y, color,ptsize):
    point_adds = product(range(-ptsize,ptsize), range(-ptsize,ptsize))
    for pt in point_adds:
        try:
            frame[x+pt[0],y+pt[1]] = color
        except IndexError:
            pass
    return frame



def make_bob_vids_trial(df_row,labels,pwin,cluster_key,base_path,vid_path,side,tdown,sdown):
    n_k = len(np.unique(labels[cluster_key]))

    # cols, col_map = make_colormap(10)
    font = cv2.FONT_HERSHEY_DUPLEX
    fontsize = 2
    cm = pylab.get_cmap('jet')
    cols = []
    for n in range(n_k):
        col = 255*np.array(mpc.to_rgb(cm(n/n_k)))
        col = col.astype('int64')
        cols.append(col)
    cols.reverse()

    labels_rows = labels_df_from_df_row(labels,df_row)
    cluster_vals = labels_rows[cluster_key].tolist()

    dlc_file = dlc_file_from_row(vid_path,side,df_row)
    pts = pd.read_hdf(dlc_file)
    pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]
    eye_x = pts['LEye x']
    eye_y = pts['LEye y']

    out_path = make_path(base_path,df_row)
    vidfile = vidname_from_row(vid_path,side,df_row)
    vidname = os.path.split(vidfile)[-1]

    vid = cv2.VideoCapture(vidfile)
    total_frames_1 = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vid.get(cv2.CAP_PROP_FPS)
    frame_width = int(vid.get(3)/sdown)
    frame_height = int(vid.get(4)/sdown)

    out = cv2.VideoWriter(os.path.join(out_path,vidname),cv2.VideoWriter_fourcc('M','J','P','G'),
                            fps/tdown,(frame_width,frame_height))

    frame_range = int(np.ceil(fps*pwin*2)) #number of frames in movement
    mov_frames = df_row['%s_windows' % side] + int(pwin*2*df_row['fps']) + int(df_row['%s_Start' % side])
    mov_frames = np.append(mov_frames,10000)
    mf_idx = 0
    for idx in range(total_frames_1):
        vid.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame_1 = vid.read()

        frame_1 = cv2.resize(frame_1,(frame_width,frame_height))

        if idx>frame_range:
            if idx==mov_frames[mf_idx]:
                frame_1 = draw_points(frame_1, (eye_y[idx-frame_range:idx]/sdown).astype('int'),
                                        (eye_x[idx-frame_range:idx]/sdown).astype('int'),
                                    cols[cluster_vals[mf_idx]],3)
                cv2.putText(frame_1, 'movement cluster %d' % (cluster_vals[mf_idx]+1), (int(frame_width*(1/2)),int(frame_height/2.5)), font, fontsize, tuple(cols[cluster_vals[mf_idx]].tolist()), 1, cv2.LINE_AA)
                mf_idx += 1
                for i in range(int(df_row['fps']/2)):
                    out.write(frame_1)
            elif idx==int(df_row['%s_Start' % side]):
                frame_1 = draw_points(frame_1, (eye_y[idx-frame_range:idx]/sdown).astype('int'),
                                        (eye_x[idx-frame_range:idx]/sdown).astype('int'),
                                    np.array([255,255,255]),3)
                cv2.putText(frame_1, 'trial start', (int(frame_width*(1/2)),int(frame_height/2.5)), font, fontsize, tuple([0,0,0]), 1, cv2.LINE_AA)
                for i in range(int(df_row['fps']/2)):
                    out.write(frame_1)
            elif idx==int(df_row['%s_Jump' % side]):
                frame_1 = draw_points(frame_1, (eye_y[idx-frame_range:idx]/sdown).astype('int'),
                                        (eye_x[idx-frame_range:idx]/sdown).astype('int'),
                                    np.array([255,255,255]),3)
                cv2.putText(frame_1, 'jump start', (int(frame_width*(1/2)),int(frame_height/2.5)), font, fontsize, tuple([0,0,0]), 1, cv2.LINE_AA)
                for i in range(int(df_row['fps']/2)):
                    out.write(frame_1)
            elif idx==int(df_row['%s_End' % side]):
                frame_1 = draw_points(frame_1, (eye_y[idx-frame_range:idx]/sdown).astype('int'),
                                        (eye_x[idx-frame_range:idx]/sdown).astype('int'),
                                    np.array([255,255,255]),3)
                cv2.putText(frame_1, 'jump end', (int(frame_width*(1/2)),int(frame_height/2.5)), font, fontsize, tuple([0,0,0]), 1, cv2.LINE_AA)
                for i in range(int(df_row['fps']/2)):
                    out.write(frame_1)
            else:
                frame_1 = draw_points(frame_1, (eye_y[idx-frame_range:idx]/sdown).astype('int'),
                                        (eye_x[idx-frame_range:idx]/sdown).astype('int'),
                                    np.array([255,255,255]),3)
                out.write(frame_1)
        else:
            if idx==int(df_row['%s_Start' % side]):
                frame_1 = draw_points(frame_1, (eye_y[idx-frame_range:idx]/sdown).astype('int'),
                                        (eye_x[idx-frame_range:idx]/sdown).astype('int'),
                                    np.array([255,255,255]),3)
                cv2.putText(frame_1, 'trial start', (int(frame_width*(1/2)),int(frame_height/2.5)), font, fontsize, tuple([0,0,0]), 1, cv2.LINE_AA)
                for i in range(int(df_row['fps']/2)):
                    out.write(frame_1)
            else:
                frame_1 = draw_points(frame_1, (eye_y[:idx]/sdown).astype('int'),
                                        (eye_x[:idx]/sdown).astype('int'),
                                        np.array([255,255,255]),3)
                out.write(frame_1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    vid.release()
    cv2.destroyAllWindows()

    print('created ' + vidname)



def make_bob_vids(df,labels,idxs,cluster_key,base_path,vid_path,side,tdown,sdown):
    N_K = len(np.unique(labels[cluster_key]))
    df = df.iloc[idxs]
    df.reset_index(inplace=True,drop=True)
    n_k = len(np.unique(labels[cluster_key]))
    font = cv2.FONT_HERSHEY_DUPLEX
    fontsize = 1.5
    cm = pylab.get_cmap('jet')
    mvmnt_label = ['no movement']
    for i in range(n_k):
        mvmnt_label.append('movement %d' % i)

    for index, row in df.iterrows():
        out_path = make_path(base_path,row)
        vidfile = vidname_from_row(vid_path,side,row)
        vidname = os.path.split(vidfile)[-1]
        dlc_file = dlc_file_from_row(vid_path,side,row)
        pts = pd.read_hdf(dlc_file)
        pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]
        eye_x = pts['LEye x'][int(row['Side_Start']):int(row['Side_End'])]
        eye_y = pts['LEye y'][int(row['Side_Start']):int(row['Side_End'])]
        col_labels = np.zeros((len(eye_x))).astype('int')
        st_frame = row[side + '_windows']
        labels = labels_df_from_df_row(labels,row)
        cluster_vals = labels[cluster_key].tolist()
        print('cluster vals: ',cluster_vals)

        vid = cv2.VideoCapture(vidfile)
        total_frames_1 = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vid.get(cv2.CAP_PROP_FPS)
        frame_width = int(vid.get(3)/sdown)
        frame_height = int(vid.get(4)/sdown)

        frame_range = int(np.ceil(fps*0.5)) #number of frames in movement (1000ms window instead of 800ms here)
        for st,c in zip(st_frame,cluster_vals):
            # st = st+frame_range
            vid_frames = np.arange(st,st+frame_range,1).astype('int')
            col_labels[vid_frames] = c+1

        out = cv2.VideoWriter(os.path.join(out_path,vidname),cv2.VideoWriter_fourcc('M','J','P','G'),
                              fps/tdown,(frame_width,frame_height))

        for idx in np.arange(int(row['Side_Start']),int(row['Side_End']),1):
            vid.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame_1 = vid.read()

            frame_1 = cv2.resize(frame_1,(frame_width,frame_height))

            if idx<len(eye_x):
                col = 255*np.array(mpc.to_rgb(cm(col_labels[idx]/N_K)))
                col = col.astype('int64')
                if idx>frame_range:
                    frame_1 = draw_points(frame_1, (eye_y[idx-frame_range:idx]/sdown).astype('int'),
                                          (eye_x[idx-frame_range:idx]/sdown).astype('int'),
                                      col,3)
                else:
                    frame_1 = draw_points(frame_1, (eye_y[:idx]/sdown).astype('int'),
                                          (eye_x[:idx]/sdown).astype('int'),
                                          col,3)
                cv2.putText(frame_1, mvmnt_label[col_labels[idx]], (int(frame_width*(2/3)),int(frame_height/4)), font, fontsize, tuple(col.tolist()), 1, cv2.LINE_AA)
            out.write(frame_1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out.release()
        vid.release()
        cv2.destroyAllWindows()
        
        print('created ' + vidname)


def plot_tail_distance(df_all,tail_pt,t_sec,condition,side,n_examples,plot_params,save_pdf,pp):
    
    ### make plots of the tail movements for individual animals, plus averages across time for individual/group data
    
    conds = np.unique(df_all[condition])
    dists = np.unique(df_all['distance_DLC'])
    anis = np.unique(df_all['subject'])
    fig, axs = plt.subplots(1,len(dists),figsize=(5*len(dists),5))

    for c,cond in enumerate(conds):
        for d,dist in enumerate(dists):
            temp_df = remove_aborts(df_all)
            temp_df = temp_df[(temp_df[condition]==cond)&(temp_df['distance_DLC']==dist)&(temp_df['success']==1)]
            temp_df.reset_index(inplace=True,drop=True)

            tail_dists = np.empty((int(df_all['fps'].iloc[0]*t_sec),temp_df.shape[0]))
            tail_dists[:] = np.nan
            for index,row in temp_df.iterrows():
                fr_dur = int(t_sec*row['fps'])
                if len(row[side + ' TailBase x'])<=fr_dur:
                    idxs = np.arange(2,len(row[side + ' TailBase x'])-1)
                else:
                    idxs = np.arange(2,fr_dur-2)
                tail_x = np.diff(row[side + ' ' + tail_pt + ' x']-row[side + ' TailBase x'])/row[side + '_pixpercm']
                tail_y = np.diff(row[side + ' ' + tail_pt + ' y']-row[side + ' TailBase y'])/row[side + '_pixpercm']
                tail_dists[idxs,index] = np.flip((np.sqrt(tail_x**2 + tail_y**2))[idxs])#/(1/row['fps']))[idxs]
            ax = axs[d]
            ax.plot(np.nanmean(tail_dists,axis=1)[2:],'-',color=plot_params['cond_col'][c],label=cond)
            ax.set_title('%s distance=%scm' % (cond,dist))
            ax.set_xlim(0,300)
            ax.set_xlabel('time before jump (s)')
            ax.set_ylabel('tail distance (cm)')
            ax.set_ylim(0,2)
            ax.set_yticks([0,1,2])
            ax.set_xticks(np.arange(0,360,60))
            ax.set_xticklabels(-np.arange(t_sec+1))
            ax = xy_axis(ax)
    ax.legend(fontsize=7)
    fig.suptitle('all animals')
    fig.tight_layout()
    if save_pdf:
        pp.savefig(fig)
        plt.close(fig)
    print('finished plotting group averages')

    for a,ani in enumerate(anis):
        fig, axs = plt.subplots(len(conds)*2,n_examples,figsize=(n_examples*5,len(conds)*10))
        for c,cond in enumerate(conds):
            temp_df = remove_aborts(df_all)
            temp_df = temp_df[(temp_df[condition]==cond)&(temp_df['subject']==ani)&(temp_df['success']==1)]
            temp_df.reset_index(inplace=True,drop=True)
            ntr = temp_df.shape[0]
            idxs = np.random.randint(0, high=ntr, size=n_examples, dtype=int)
            for i,idx in enumerate(idxs):
                row = temp_df.iloc[idx]
                # tail_dxy  = np.diff(tip - base)
                # tail_dist = np.sqrt(tail_dxy[0,:]**2 + tail_dxy[1,:]**2)
    #             print(row['expdate'],row['subject'],row['trial'])

                tail_x = np.diff(row[side + ' ' + tail_pt + ' x']-row[side + ' TailBase x'])/row[side + '_pixpercm']
                tail_y = np.diff(row[side + ' ' + tail_pt + ' y']-row[side + ' TailBase y'])/row[side + '_pixpercm']
                tail_dist = np.sqrt(tail_x**2 + tail_y**2)
                tail_like = row[side + ' ' + tail_pt + ' likelihood']
                ax = axs[c,i]
                ax.plot(np.flip(tail_x[2:-2]),label='tail x')
                ax.plot(np.flip(tail_y[2:-2]),label='tail y')
                ax.plot(np.flip(tail_dist[2:-2]),label='distance_DLC')
                ax.plot(np.flip(tail_like[2:-2])+9,label='likelihood')
                ax.plot([0,300],[10,10],':',color=[0.5,0.5,0.5])
                ax.plot([0,300],[9,9],':',color=[0.5,0.5,0.5])
                ax.set_xlabel('time before jump (s)')
                ax.set_ylabel('tail distance (cm)')
                ax.set_ylim(-10,10.5)
                ax.set_xlim(0,300)
                ax.set_yticks([-10,-5,0,5,10])
                ax.set_xticks(np.arange(0,360,60))
                ax.set_xticklabels(-np.arange(t_sec+1))
                ax.set_title(cond)
                ax = xy_axis(ax)

                ax = axs[c+len(conds),i]
                ax.plot(row[side + ' TailTip x'][2:-2]/row[side + '_pixpercm'],row[side + ' TailTip y'][2:-2]/row[side + '_pixpercm'],'c-',label='tail tip')
                ax.plot(row[side + ' MidTail x'][2:-2]/row[side + '_pixpercm'],row[side + ' MidTail y'][2:-2]/row[side + '_pixpercm'],'b-',label='tail mid')
                ax.plot(row[side + ' TailBase x'][2:-2]/row[side + '_pixpercm'],row[side + ' TailBase y'][2:-2]/row[side + '_pixpercm'],'k-',label='tail base')
                ax.set_xlim(int(180/row[side + '_pixpercm']),int(720/row[side + '_pixpercm']))
                ax.set_ylim(int(0/row[side + '_pixpercm']),int(540/row[side + '_pixpercm']))

                ax.set_xlabel('distance (cm)')
                ax.set_ylabel('distance (cm)')
                ax = xy_axis(ax)
        axs[len(conds)+len(conds)-1,n_examples-1].legend(fontsize=10,loc=4)
        axs[len(conds)-1,n_examples-1].legend(fontsize=10,loc=4)
        fig.suptitle(ani)
        fig.tight_layout()
        if save_pdf:
            pp.savefig(fig)
            plt.close(fig)

        fig, axs = plt.subplots(1,len(dists),figsize=(5*len(dists),5))
        for c,cond in enumerate(conds):
            for d,dist in enumerate(dists):
                temp_df = remove_aborts(df_all)
                temp_df = temp_df[(temp_df[condition]==cond)&(temp_df['distance_DLC']==dist)&(temp_df['subject']==ani)&(temp_df['success']==1)]
                temp_df.reset_index(inplace=True,drop=True)

                tail_dists = np.empty((int(row['fps']*t_sec),temp_df.shape[0]))
                tail_dists[:] = np.nan
                for index,row in temp_df.iterrows():
                    fr_dur = int(t_sec*row['fps'])
                    if len(row[side + ' TailBase x'])<fr_dur:
                        idxs = np.arange(0,len(row[side + ' TailBase x'])-1)
                    else:
                        idxs = np.arange(0,fr_dur-1)
                    tail_x = np.diff(row[side + ' ' + tail_pt + ' x']-row[side + ' TailBase x'])/row[side + '_pixpercm']
                    tail_y = np.diff(row[side + ' ' + tail_pt + ' y']-row[side + ' TailBase y'])/row[side + '_pixpercm']
                    tail_dists[idxs,index] = np.flip((np.sqrt(tail_x**2 + tail_y**2))[idxs])#/(1/row['fps']))[idxs]
                
                ax = axs[d]
                ax.plot(np.nanmean(tail_dists,axis=1)[2:],'-',color=plot_params['cond_col'][c],label=cond)
                ax.set_title('%s distance=%scm' % (cond,dist))
                ax.set_xlim(0,300)
                ax.set_xlabel('time before jump (s)')
                ax.set_ylabel('tail distance (cm)')
                ax.set_ylim(0,5)
                ax.set_yticks(np.arange(6))
                ax.set_xticks(np.arange(0,360,60))
                ax.set_xticklabels(-np.arange(t_sec+1))
                ax = xy_axis(ax)
        ax.legend(fontsize=7)
        fig.suptitle(ani)
        fig.tight_layout()
        if save_pdf:
            pp.savefig(fig)
            plt.close(fig)
        print('finished plotting animal %d of %d' % (a+1,len(anis)))

    return