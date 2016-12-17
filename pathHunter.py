import cv2
import glob
import code
import numpy as np
import sys
from timeit import default_timer as timer
import os
from itertools import cycle
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import cPickle as pickle
import random
import collections
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from graph_tool.all import *





# /*
# ███    ███  █████  ██ ███    ██
# ████  ████ ██   ██ ██ ████   ██
# ██ ████ ██ ███████ ██ ██ ██  ██
# ██  ██  ██ ██   ██ ██ ██  ██ ██
# ██      ██ ██   ██ ██ ██   ████
# */

def main():
	dirr = sys.argv[1]

	list_of_image_paths = sorted(glob.glob(dirr +'*'))

	list_of_image_paths = [i for j, i, in enumerate(list_of_image_paths) if j not in indices_of_slices_to_be_removed]

	shape = cv2.imread(list_of_image_paths[0],-1).shape

	images = []
	for i, path in enumerate(list_of_image_paths):
		im = cv2.imread(path, -1)
		images.append(im)
	print 'Loaded ' + str(len(images)) + ' images.'

	imageArray = np.dstack(images)

	colorLists = []
	for z in xrange(imageArray.shape[2]):
		colorLists.append([c for c in np.unique(imageArray[:,:,z]) if c!=0])

	colorArray = np.dstack(colorLists)[0]

	colorSet = set(colorArray.flatten())
	colorMasterList = list(colorSet)

	for z in xrange(colorArray.shape[0]):



if __name__ == "__main__":
	main()
