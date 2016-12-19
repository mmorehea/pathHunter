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

#def markBlob(x, y):


def mouseClick(event,x,y,flags,param):
	global zChange, firstClick, image
	if event == cv2.EVENT_LBUTTONDOWN:
		print "value at " + str(x) + ", " + str(y)
		print " " + str(image[y, x])
		if firstClick and image[y, x] == 65000:
			print "found first blob, continue"
			firstClick = not firstClick
			zChange = True
		else:
			print "got next blob, coloring"
			zChange = True

def findBBDimensions(listofpixels):
	if len(listofpixels) == 0:
		return None
	else:
		xs = [x[0] for x in listofpixels]
		ys = [y[1] for y in listofpixels]

		minxs = min(xs)
		maxxs = max(xs)

		minys = min(ys)
		maxys = max(ys)

		dx = max(xs) - min(xs)
		dy = max(ys) - min(ys)


		return [minxs, maxxs, minys, maxys], [dx, dy]

def findCentroid(listofpixels):
	if len(listofpixels) == 0:
		return (0,0)
	rows = [p[0] for p in listofpixels]
	cols = [p[1] for p in listofpixels]
	try:
		centroid = int(round(np.mean(rows))), int(round(np.mean(cols)))
	except:
		# code.interact(local=locals())
		centroid = (0,0)
	return centroid

def getMeasurements(blob, shape):
	img = np.zeros(shape, np.uint16)
	img[zip(*blob)] = 1
	per = []
	for p in blob:
		x = p[0]
		y = p[1]
		q = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]

		edgePoint = False
		for each in q:
			try:
				if img[each] == 0:
					edgePoint = True
			except IndexError:
				edgePoint = True
		if edgePoint:
			per.append(p)
	return len(blob), len(per)

def testOverlap(setofpixels1, setofpixels2):

	set_intersection = setofpixels1 & setofpixels2

	set_union = setofpixels1 | setofpixels2

	percent_overlap = float(len(set_intersection)) / len(set_union)

	return percent_overlap

def orderByPercentOverlap(blobs, reference):
	overlapList = []
	for blob in blobs:
		overlapList.append((testOverlap(set(reference),set(blob)), blob))


	overlapList = sorted(overlapList,key=lambda o: o[0])[::-1]
	orderedBlobs = [l[1] for l in overlapList]
	overlapVals = [l[0] for l in overlapList]

	return orderedBlobs, overlapVals

def waterShed(blob, shape):
	img = np.zeros(shape, np.uint16)
	img[zip(*blob)] = 99999

	D = ndimage.distance_transform_edt(img)
	mindist = 7
	labels = [1,2,3,4]
	while len(np.unique(labels)) > 3:
		mindist += 1
		localMax = peak_local_max(D, indices=False, min_distance=mindist, labels=img)

		markers = ndimage.label(localMax, structure=np.ones((3,3)))[0]
		labels = watershed(-D, markers, mask=img)

	subBlobs = []
	for label in np.unique(labels):
		if label == 0:
			continue
		ww = np.where(labels==label)
		bb = zip(ww[0], ww[1])
		subBlobs.append(bb)
	# code.interact(local=locals())
	try:
		return subBlobs, zip(np.where(localMax==True)[0],np.where(localMax==True)[1])[0]
	except IndexError:
		return subBlobs, 0

def findNearest(img, startPoint):
	directions = cycle([[0,1], [1,1], [1,0], [1,-1], [0,-1], [-1,-1], [-1,0], [-1,1]])
	increment = 0
	cycleCounter = 0
	distance = [0,0]

	if img[startPoint] > 0:
		return startPoint

	while True:
		direction = directions.next()

		for i in [0,1]:
			if direction[i] > 0:
				distance[i] = direction[i] + increment
			elif direction[i] < 0:
				distance[i] = direction[i] - increment
			else:
				distance[i] = direction[i]

		checkPoint = (startPoint[0] + distance[0],startPoint[1] + distance[1])

		cycleCounter += 1
		if cycleCounter % 8 == 0:
			increment += 1

		# print cycleCounter

		try:
			if img[checkPoint] > 0:
				break
		except:
			#code.interact(local=locals())
			break

	return checkPoint

def distance(point1, point2):
	return ((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)**0.5

def display(blob):

	img = np.zeros(shape, np.uint16)
	for pixel in blob:
		img[pixel] = 99999

	cv2.imshow(str(random.random()),img)
	cv2.waitKey()

def transformBlob(blob, displacement):
	dx, dy = displacement

	transformedBlob = []
	for point in blob:
		newPoint = (point[0] + dx, point[1] + dy)
		transformedBlob.append(newPoint)

	return transformedBlob

def shapeMatch(blob1, blob2, shape):
	img1 = np.zeros(shape, np.uint8)
	img2 = img1.copy()
	img1[zip(*blob1)] = 99999
	img2[zip(*blob2)] = 99999

	im, contours, hierarchy = cv2.findContours(img1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cnt1 = contours[0]
	im, contours, hierarchy = cv2.findContours(img2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cnt2 = contours[0]

	match = cv2.matchShapes(cnt1, cnt2, 1, 0)
	return match

def resetStats(currentBlob, centroid1, blob2, freq, organicWindow, displacementBuffer):

	centroid2 = findCentroid(blob2)
	overlap = testOverlap(set(currentBlob), set(blob2))
	coverage = freq / float(len(organicWindow))
	freq2 = len(set(currentBlob) & set(blob2))
	coverage2 = freq2 / float(len(blob2))

	# dx = centroid2[0] - centroid1[0]
	# dy = centroid2[1] - centroid1[1]
	# displacementBuffer.append((dx,dy))
	# if len(displacementBuffer) > 5:
	# 	del displacementBuffer[0]
	# #collect average displacement buffer
	# dxs = [x[0] for x in displacementBuffer]
	# dys = [x[1] for x in displacementBuffer]
	# avgDisplacement_last5 = (float(sum(dxs))/5, float(sum(dys))/5)

	return centroid2, overlap, coverage, freq2, coverage2, displacementBuffer

def trackProcess(startBlob, imageArray, z):
	image = imageArray[:,:,z]
	box, dimensions = findBBDimensions(startBlob)
	color1 = image[startBlob[0]]
	centroid1 = findCentroid(startBlob)
	startZ = z
	process = [startBlob]
	shape = image.shape

	currentBlob = startBlob

	zspace = 0
	d = 0
	skip = 0
	splitCount = 0
	terminate = False
	splitRecent = False
	splitList = []
	displacementBuffer = []
	while terminate == False:
		zspace += 1
		print zspace
		blobsfound = []
		try:
			image2 = imageArray[:,:,z+zspace]
		except:
			terminate = True
			s = '0'
			continue

		window = image2[box[0]:box[1], box[2]:box[3]]
		organicWindow = image2[zip(*currentBlob)]
		frequency = collections.Counter(organicWindow).most_common()

		#check for blackness
		if frequency[0][0] == 0 and len(frequency) == 1:
			if d > 10:
				terminate = True
				while d > 0:
					del process[-1]
					d -= 1
				continue
			else:
				process.append([])
				d += 1
				continue

		# find largest color that is not black
		for each in frequency:
			if each[0] == 0:
				continue
			clr, freq = each
			break

		# get those pixels that are that color

		# figure out features that describe realtionship between shapes
		q = np.where(image2 == clr)
		blob2 = [(i[0],i[1], z+zspace) for i in q]

		centroid2 = findCentroid(blob2)
		overlap = testOverlap(set(currentBlob), set(blob2))
		coverage = freq / float(len(organicWindow))
		freq2 = len(set(currentBlob) & set(blob2))
		coverage2 = freq2 / float(len(blob2))

		print str(coverage)
		#thresholds for deciding to add this blob
		if coverage > 0.7:
			process.append(blob2)
			print blob2
		else:
			terminate = True


	return zspace, process, color1

# /*
# ███    ███  █████  ██ ███    ██
# ████  ████ ██   ██ ██ ████   ██
# ██ ████ ██ ███████ ██ ██ ██  ██
# ██  ██  ██ ██   ██ ██ ██  ██ ██
# ██      ██ ██   ██ ██ ██   ████
# */

################################################################################
# SETTINGS
minimum_process_length = 0
write_images_to = 'littleresult/'
write_pickles_to = 'pickles/object'
trace_objects = True
build_resultStack = True
load_stack_from_pickle_file = False
indices_of_slices_to_be_removed = []
################################################################################

def main():
	global zChange
	global firstClick
	global image
	zChange = 0
	firstClick = True
	blobSet = {}

	dirr = sys.argv[1]
	#collecting Tiffs
	list_of_image_paths = sorted(glob.glob(dirr +'*'))
	list_of_image_paths = [i for j, i, in enumerate(list_of_image_paths) if j not in indices_of_slices_to_be_removed]
	zMax = len(list_of_image_paths)
	shape = cv2.imread(list_of_image_paths[0],-1).shape
	images = []
	for i, path in enumerate(list_of_image_paths):
		im = cv2.imread(path, -1)
		images.append(im)
	print 'Loaded ' + str(len(images)) + ' images.'
	imageArray = np.dstack(images)

	# finds all unique colors inside of 3D volume
	colorList = []
	for z in xrange(imageArray.shape[2]):
		colorList.extend([c for c in np.unique(imageArray[:,:,z]) if c!=0])
	colorList = list(set(colorList))

	z = 0

	image = imageArray[:,:,z]
	colorVals = [c for c in np.unique(image) if c!=0]

	blobToTrack = 6506
	startBlob = zip(np.where(image == blobToTrack)[0], np.where(image == blobToTrack)[1])
	image[np.where(image == blobToTrack)] = 65000

	cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	cv2.setMouseCallback('image', mouseClick)

	while(1):
		if zChange:
			zDiff, process, color1 = trackProcess(startBlob, imageArray, z)
			blobSet[blobToTrack] = process
			z += zDiff
			if z >= zMax:
				print blobSet
				print len(blobSet[6506])
				break
			image = imageArray[:,:,z]
		cv2.imshow('image',image)

		k = cv2.waitKey(1) & 0xFF
		if k == 27:
			break

	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
