import argparse
import random
import os

from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data')
parser.add_argument('--output_dir', default='preprocessed_data')

def resize_and_save(filename, output_dir, size=64):
    image = Image.open(filename)
    image = image.resize((size,size))
    image.save(os.path.join(output_dir, filename.split('/')[-1]))
    
def make_directory(dir_name, folder_name):
    path = os.path.join(dir_name, folder_name)
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == "__main__":    
    args = parser.parse_args()
    
    train_data_dir = os.path.join(args.data_dir, "train_signs")
    test_data_dir = os.path.join(args.data_dir, "test_signs")    
    
    filenames = os.listdir(train_data_dir)
    filenames = [os.path.join(train_data_dir, f) for f in filenames if 'DS_Store' not in f]
    
    random.seed(230)
    random.shuffle(filenames)

    split = int(0.9 * len(filenames))
    train_filenames = filenames[:split]
    val_filenames = filenames[split:]
    test_filenames = [os.path.join(test_data_dir,f) for f in os.listdir(test_data_dir) if 'DS_Store' not in f]
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    # preprocess train
    make_directory(args.output_dir, "train_signs")
    print("-- Processing training data, saving preprocessed data to :" + os.path.join(args.output_dir, "train_signs"))
    for filename in tqdm(train_filenames):
        resize_and_save(filename, os.path.join(args.output_dir, "train_signs"))
    
    # preprocess val
    print("-- Processing validation data, saving preprocessed data to :" + os.path.join(args.output_dir, "val_signs"))
    make_directory(args.output_dir, "val_signs")
    for filename in tqdm(val_filenames):
        resize_and_save(filename, os.path.join(args.output_dir, "val_signs"))
    
    # preprocess test
    print("-- Processing testing data, saving preprocessed data to :" + os.path.join(args.output_dir, "test_signs"))
    make_directory(args.output_dir, "test_signs")
    for filename in tqdm(test_filenames):
        resize_and_save(filename, os.path.join(args.output_dir, "test_signs"))
        
    print("-- DONE!")
    