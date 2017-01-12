# -*- coding: utf-8 -*-
import argparse
import cv2
import glob
import code
import numpy as np
import sys
from timeit import default_timer as timer
import os
from itertools import cycle
from itertools import combinations
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import cPickle as pickle
import random
import collections
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
from mpl_toolkits.mplot3d import Axes3D
from skimage import data
from skimage.feature import match_template
import math
import json
import tifffile

def buildResultStack(start, write_images_to, write_pickles_to, maskPaths, maskShape):
	picklePaths = sorted(glob.glob(write_pickles_to + '*.p'))

	pickles = [pickle.load(open(path,'rb')) for path in picklePaths]

	for z in xrange(len(maskPaths)):
		resultImg = np.zeros(maskShape, np.uint16)
		maskImg = cv2.imread(maskPaths[z], -1)

		for process, color in pickles:
			if z in process.keys():
				for each in process[z]:
					if each == None:
						continue
					if str(each).isdigit():
						resultImg[np.where(maskImg==each)] = color
					else:
						resultImg[zip(*each)] = color

		tifffile.imsave(write_images_to + maskPaths[z][maskPaths[z].index('/')+1:], resultImg)
		print '\n'
		print 'Building result stack ' + str(z+1) + '/' + str(len(maskPaths))
		end = timer()
		print(end - start)
		print '\n'
# ███    ███  █████  ██ ███    ██
# ████  ████ ██   ██ ██ ████   ██
# ██ ████ ██ ███████ ██ ██ ██  ██
# ██  ██  ██ ██   ██ ██ ██  ██ ██
# ██      ██ ██   ██ ██ ██   ████
def main():
	################################################################################
	# SETTINGS
	minimum_process_length = 100 # Be careful not to set this too high because there may be small chains that can be merged manually with larger chains to complete them
	write_images_to = 'build/'
	write_pickles_to = 'picklecrop/object'
	################################################################################
	# Profiling:
	# python -m cProfile -o output pathHunter.py littlecrop/
	# python runsnake.py output

	# Get list of colors to use in the result stack
	masterColorList = pickle.load(open('masterColorList.p','rb'))
	print "Loading data..."
	maskFolderPath = sys.argv[1]
	emFolderPath = sys.argv[2]
	maskPaths =  sorted(glob.glob(maskFolderPath +'*'))
	emPaths = sorted(glob.glob(emFolderPath +'*'))
	maskImages = cv2.imread(maskPaths[0], -1)

	maskShape = maskImages.shape

	startTime = timer()
	print "Beginning build..."
	buildResultStack(startTime, write_images_to, write_pickles_to, maskPaths, maskShape)

if __name__ == "__main__":
	main()
