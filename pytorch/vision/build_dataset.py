"""Split the SIGNS dataset into train/val/test and resize images to 64x64.

The SIGNS dataset comes into the following format:
    train_signs/
        0_IMG_5864.jpg
        ...
    test_signs/
        0_IMG_5942.jpg
        ...

Original images have size (3024, 3024).
Resizing to (64, 64) reduces the dataset size from 1.16 GB to 4.7 MB, and loading smaller images
makes training faster.

We already have a test set created, so we only need to split "train_signs" into train and val sets.
Because we don't have a lot of images and we want that the statistics on the val set be as
representative as possible, we'll take 20% of "train_signs" as val set.
"""

# 总结一下做法
# 1. 从已有的train_set中获取全部图像文件名的列表
# 2. 将这个列表进行打乱，取80%作为train_set, 取20%作为val_set
# 3. 获得一个filename的字典，其中包含train_set, val_set和test_set中所有图片的列表
# 4. 创建一个output_dir的目录，再往其中创建3个子目录，用来存储预处理后的图像
# 5. 用PIL库中的API对图像进行处理，然而存在对应的子目录中

import argparse
import random
import os

from PIL import Image
from tqdm import tqdm

SIZE = 64

# 这个是用来处理python的命令行参数的
# default就是缺省值，help应该就是帮助文档
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/SIGNS', help="Directory with the SIGNS dataset")
parser.add_argument('--output_dir', default='data/64x64_SIGNS', help="Where to write the new data")


# 使用PIL库进行resize处理，可以有效降低我们图像数据的大小
def resize_and_save(filename, output_dir, size=SIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    image.save(os.path.join(output_dir, filename.split('/')[-1]))   # 创建新的图像之后我们将其保存在我们的目录下


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the data directories
    train_data_dir = os.path.join(args.data_dir, 'train_signs')
    test_data_dir = os.path.join(args.data_dir, 'test_signs')

    # Get the filenames in each directory (train and test)
    filenames = os.listdir(train_data_dir)  
    # 获取每一个训练集中图片的路径
    filenames = [os.path.join(train_data_dir, f) for f in filenames if f.endswith('.jpg')]  # 这里用到了python中变量重用的思想

    # 同理，获取每一个测试集中图片的路径
    test_filenames = os.listdir(test_data_dir)
    test_filenames = [os.path.join(test_data_dir, f) for f in test_filenames if f.endswith('.jpg')]

    # Split the images in 'train_signs' into 80% train and 20% val
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(230)
    filenames.sort()
    random.shuffle(filenames)   # random.shuffle的作用是将filename中的内容全部打乱

    # 将文件全部打乱以后我们就可以从中取得train和val的图片集了
    # 始终记得要把我们的训练集分为train和val两个集合
    split = int(0.8 * len(filenames))
    train_filenames = filenames[:split]
    val_filenames = filenames[split:]

    # filename是一个字典，其键值对中的值是一个列表。列表的内容则是所有图片文件名
    filenames = {'train': train_filenames,
                 'val': val_filenames,
                 'test': test_filenames}

    # 看来我们还要创建output_dir目录，然后再在其后面创建子目录
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    # Preprocess train, val and test
    for split in ['train', 'val', 'test']:
        # 在ouput_dir中创建子目录在存储你预处理后的图像
        # 1. 创建子目录路径的字符串 2. 判断这个目录是否存在 3. 将row data文件夹中的数据预处理后放入你创建的路径
        # 注意这个join会创建子目录
        output_dir_split = os.path.join(args.output_dir, '{}_signs'.format(split)) 
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))

        print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
        for filename in tqdm(filenames[split]):     # tqdm没什么用，只是给你的循环添加一个进度条而已
            resize_and_save(filename, output_dir_split, size=SIZE)

    print("Done building dataset")
