import os
import shutil

def copy_files():
    src = "/vol/bitbucket/rp22/sdspeech/data/audiocaps/dataset_full/test"
    
    dest = "/vol/bitbucket/rp22/sdspeech/evaluation/audiocaps/data/test"
    
    # Make sure the destination directory exists; create it if it doesn't
    if not os.path.exists(dest):
        os.makedirs(dest)

    # Loop through all files in the source directory
    for filename in os.listdir(src):
        # If the file ends with .wav, copy it to the destination directory
        if filename.endswith('.wav'):
            src_path = os.path.join(src, filename)
            dest_path = os.path.join(dest, filename)
            shutil.copy(src_path, dest_path)
            
copy_files()