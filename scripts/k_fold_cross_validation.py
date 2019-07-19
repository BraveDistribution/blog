# This script is used for creating the dataset into K-folds,
# when each fold is used k-1 times in training and 1 in testing phase

# Let's have this structure:
# data/0/
# ------img01.jpg
# ------img02.jpg
# data/1/
# ------img03.jpg
# ------img04.jpg

# split it into K-folds output
# /data/first_loop/train (consisting of k-1 folds)
# /data/first_loop/test (consisting of 1 fold)

import argparse
import itertools
import logging
import numpy as np
import os
import shutil
import sys
import traceback

from pathlib import Path

from sklearn.model_selection import StratifiedKFold


def create_empty_directory_for_all_files(path):
    try:
        os.mkdir(path)
    except OSError:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("*** print_tb:")
        traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
        print("*** print_exception:")
        # exc_type below is ignored on 3.5 and later
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  limit=2, file=sys.stdout)
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)


def create_empty_directory_for_all_classes(fold_dir, classes):
    for class_label in classes:
        create_empty_directory_for_all_files(Path(fold_dir + '/train/'))
        create_empty_directory_for_all_files(Path(fold_dir + '/train/' + str(class_label)))
        create_empty_directory_for_all_files(Path(fold_dir + '/test/'))
        create_empty_directory_for_all_files(Path(fold_dir + '/test/' + str(class_label)))

logging.getLogger().setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(
    description="Split dataset folder into folds for cross validation",
    formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('folds_count', metavar='folds_count', type=int, help='Number of folds')
parser.add_argument('path', metavar='path', type=str, help='Path to dataset')
parser.add_argument('target_dir', metavar='target_dir', type=str, help='folder where folds should be saved')

args = parser.parse_args()

folds_count = args.folds_count
path = Path(args.path)
target_dir = Path(args.target_dir)

logging.debug("folds_count = %s, path = %s" % (str(folds_count), str(path)))

# path to the folders (images of classes)
folders = [Path((str(path) + '/' + i)) for i in os.listdir(str(path)) if os.path.isdir(str(path) + '/' + i)]

logging.debug("%d classes have been found" % len(folders))
logging.debug("Following classes have been found: %s" % folders)

# file names
names = {}

for index, folder in enumerate(folders):
    names[index] = [str(folder) + '/' + image_name for image_name in os.listdir(folder)]
    logging.debug("class %d has %d items" % (index, len(names[index])))

# X holds the FULL PATH to images and Y holds labels (0,1,2 ...)
X = np.array([item for sublist in names for item in names[sublist]])
Y = np.array(list(itertools.chain.from_iterable([[int(item)] * len(names[item]) for item in names])))

kf = StratifiedKFold(n_splits=folds_count, shuffle=True)

# using Stratified K-FOld cross validation from scikit learn to divide up data into folds_count folds.
create_empty_directory_for_all_files(target_dir)
for fold_index, (train_indexes, test_indexes) in enumerate(kf.split(X, Y)):
    fold_dir = str(target_dir) + "/k_" + str(fold_index)
    create_empty_directory_for_all_files(fold_dir)
    create_empty_directory_for_all_classes(fold_dir, names.keys())
    # copy files to specifing train and test directories
    for train_index in train_indexes:
        shutil.copyfile(X[train_index],
                        Path(fold_dir + '/train/' + str(Y[train_index]) + '/' + X[train_index].split('/')[-1]))
    for test_index in test_indexes:
        shutil.copyfile(X[test_index], Path(fold_dir + '/test/' + str(Y[test_index]) + '/' + X[test_index].split('/')[-1]))
