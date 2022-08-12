import sys, os,fnmatch
import deeplabcut as dlc
# import tensorflow as tf

# def get_immediate_subdirectories(a_dir):
#     return [name for name in os.listdir(a_dir)
#             if os.path.isdir(os.path.join(a_dir, name))]

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

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)

dlc_config_path = r'T:\jumping_revisions_dlc\jumping_revisions-Phil-2022-02-11\config.yaml'
expt_path = r'T:\jumping_revisions'
side_files = find('*SIDEcalib.avi',expt_path)
top_files = find('*TOP*.avi',expt_path)
vid_files = side_files + top_files

# dlc.analyze_videos(dlc_config_path, vid_files, save_as_csv=True)
for vid_file in vid_files:
  # print('filtering predictions')
  # dlc.filterpredictions(dlc_config_path, vid_file)
  print('making labeled videos')
  dlc.create_labeled_video(dlc_config_path, vid_files, save_frames = False)