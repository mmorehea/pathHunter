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
from skimage import data
from skimage.feature import match_template

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

def upperLeftJustify(blob):
	box, dimensions = findBBDimensions(blob)
	transformedBlob = []
	for point in blob:
		transformedPoint = (point[0] - box[0], point[1] - box[2])
		transformedBlob.append(transformedPoint)

	return transformedBlob


def trackProcess(startBlob, maskPaths, emPaths, z, shape):
	maskImage = cv2.imread(maskPaths[z], -1)
	emImage = cv2.imread(emPaths[z], -1)
	box, dimensions = findBBDimensions(startBlob)
	color1 = maskImage[startBlob[0]]
	centroid1 = findCentroid(startBlob)
	startZ = z
	process = {z:color1}
	shape = maskImage.shape

	# BLOCKING OUT
	# image[zip(*startBlob)] = 0

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
		blobsfound = []
		try:
			maskImage2 = cv2.imread(maskPaths[z+zspace], -1)
			emImage2 = cv2.imread(emPaths[z+zspace], -1)
		except:
			terminate = True
			s = '0'
			continue

		window = maskImage2[box[0]:box[1], box[2]:box[3]]
		organicWindow = maskImage2[zip(*currentBlob)]
		frequency = collections.Counter(organicWindow).most_common()

		#check for blackness
		if frequency[0][0] == 0 and len(frequency) == 1:
			if d > 10:
				terminate = True
				continue
			else:
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
		q = np.where(maskImage2 == clr)
		nextBlob = zip(q[0],q[1])
		box2, dimensions2 = findBBDimensions(nextBlob)

		blob1 = upperLeftJustify(currentBlob)
		blob2 = upperLeftJustify(nextBlob)

		emBlob = emImage[box[0]:box[1],box[2]:box[3]]

		# emBlob2 = np.zeros((dimensions2[0] + 1, dimensions2[1] + 1), np.uint8)
		# emBlob2[zip(*blob2)] = emImage2[zip(*nextBlob)]

		emBlob2 = emImage2[box2[0]:box2[1],box2[2]:box2[3]]

		if emBlob.size > emBlob2.size:
			emBlob, emBlob2 = emBlob2, emBlob

		result = match_template(emBlob2, emBlob)
		uniqueVals = np.unique(result)
		avg = float(sum(uniqueVals)) / len(uniqueVals)
		maximum = np.max(result)

		print avg, maximum
		code.interact(local=locals())

		centroid2 = findCentroid(blob2)
		overlap = testOverlap(set(currentBlob), set(blob2))
		coverage = freq / float(len(organicWindow))
		freq2 = len(set(currentBlob) & set(blob2))
		coverage2 = freq2 / float(len(blob2))
		# shapeDiff = shapeMatch(currentBlob, blob2, shape)

		# if skip < 4:
		# 	if coverage2 < 0.5:
		# 		process.append([])
		# 		skip += 1
		# 		continue
		# else:
		# 	terminate = True
		# 	skip = 0
		# 	continue

		blobsfound.append(blob2)
		skip = 0

		if terminate == False:

			newBlob = []
			for b in blobsfound:
				newBlob += b

			# shapeDiff = shapeMatch(currentBlob, newBlob, shape)

			# print str(zspace) + '. ' + str(overlap) + ' ' + str(coverage) + ' ' + str(coverage2) + ' ' + str(shapeDiff)

			#Probably need to do the stuff below when I terminate as well
			color1 = image2[newBlob[0]]
			image2[zip(*newBlob)] = 0
			process.append(newBlob)
			box,dimensions = findBBDimensions(newBlob)
			d = 0
			centroid1 = findCentroid(newBlob)
			currentBlob = newBlob
	return startZ, process, color1

# /*
# ███    ███  █████  ██ ███    ██
# ████  ████ ██   ██ ██ ████   ██
# ██ ████ ██ ███████ ██ ██ ██  ██
# ██  ██  ██ ██   ██ ██ ██  ██ ██
# ██      ██ ██   ██ ██ ██   ████
# */

def main():
	################################################################################
	# SETTINGS
	minimum_process_length = 0
	write_images_to = 'littleresult/'
	write_pickles_to = 'picklecrop/object'
	trace_objects = True
	build_resultStack = True
	load_stack_from_pickle_file = False
	indices_of_slices_to_be_removed = []
	################################################################################
	#Profiling:
	# python -m cProfile -o output pathHunter.py littlecrop/
	# python runsnake.py output

	maskFolderPath = sys.argv[1]
	emFolderPath = sys.argv[2]

	#collecting Tiffs
	maskPaths =  sorted(glob.glob(maskFolderPath +'*'))
	emPaths = sorted(glob.glob(emFolderPath +'*'))

	maskShape = cv2.imread(maskPaths[0],-1).shape
	emShape = cv2.imread(emPaths[0],-1).shape

	if len(maskPaths) != len(emPaths) or maskShape != emShape:
		print 'Error, mask and EM data do not match'
		trace_objects = False
		build_resultStack = False


	if trace_objects:
		# general setup
		chainLengths = []
		objectCount = -1

		# finds all unique colors inside of 3D volume
		# NEED TO CHANGE THIS:
		# colorList = []
		# for z in xrange(imageArray.shape[2]):
		# 	colorList.extend([c for c in np.unique(imageArray[:,:,z]) if c!=0])
		# colorList = list(set(colorList))

		# begin searching through slices

		for z in xrange(len(maskPaths)):
			###Testing###
			if z != 0:
				continue
			#############
			# get only that slice and find unique blobs
			image = cv2.imread(maskPaths[z], -1)
			colorVals = [c for c in np.unique(image) if c!=0]
			###Testing###
			colorVals = [22013, 20601, 21131, 21055]
			# 24661 @1: 24762, @9:26048, @214: 26501
			# 22013 @62: 22568 @63: 21963 @67: 22517 @68: 22971, @73: 22946
			# @258: 27384
			#############
			# blobs = []
			# for color in colorVals:
			# 	wblob = np.where(image==color)
			# 	blob = zip(wblob[0], wblob[1])
			# 	blobs.append(blob)
			# blobs = sorted(blobs, key=len)

			# with all blobs, begin searching one by one for objects
			for i, startColor in enumerate(colorVals):
				# print str(i+1) + '/' + str(len(blobs))

				wblob = np.where(image==startColor)
				startBlob = zip(wblob[0],wblob[1])

				# blacks out first blob
				start = timer()
				startZ, process, color = trackProcess(startBlob, maskPaths, emPaths, z, emShape)


				if len(process) > minimum_process_length:
					objectCount += 1
					color = colorList[objectCount]
					print '\n'
					print objectCount
					end = timer()
					print(end - start)
					print '\n'
					chainLengths.append((objectCount, color, len(process)))
					pickle.dump((startZ, process, color), open(write_pickles_to + str(objectCount) + '.p', 'wb'))


		print 'Number of chains: ' + str(len(chainLengths))
		print 'Average chain length: ' + str(float(sum([x[2] for x in chainLengths]))/len(chainLengths))
		# print s

		if os.path.exists('summary.txt'):
			os.remove('summary.txt')

		chainLengths = sorted(chainLengths)[::-1]

		with open('summary.txt','w') as f:
			for i,each in enumerate(chainLengths):
				f.write(str(chainLengths[i][0]) + ' ' + str(chainLengths[i][1]) + ' ' + str(chainLengths[i][2]) + '\n')


	if build_resultStack:

		picklePaths = sorted(glob.glob(write_pickles_to + '*.p'))

		if load_stack_from_pickle_file:
			resultArray, startO = pickle.load(open('resultArraySave.p', 'rb'))
		else:
			resultArray = np.zeros((shape[0], shape[1], len(list_of_image_paths)), np.uint16)
			startO = 0


		for o, path in enumerate(picklePaths):
			if o < startO:
				continue

			startZ, process, color = pickle.load(open(path, 'rb'))

			for z in xrange(resultArray.shape[2]):
				img = resultArray[:,:,z]

				if z < startZ:
					continue

				if z >= startZ + len(process):
					continue

				img[zip(*process[z - startZ])] = color

			pickle.dump((resultArray, o), open('resultArraySave.p,','wb'))

			print '\n'
			print 'Built object ' + str(o+1) + '/' + str(len(picklePaths))
			end = timer()
			print(end - start)
			print '\n'

		for z in xrange(resultArray.shape[2]):
			image = resultArray[:,:,z]
			cv2.imwrite(write_images_to + list_of_image_paths[z][list_of_image_paths[z].index('/')+1:], image)

		# diffarray = [(measurement[1]**2)/(4*3.1415926) - measurement[0] for measurement in measurementsList]
		#
		# ratarray = [float(measurement[1])/measurement[0] for measurement in measurementsList]

		# plt.figure(1)
		# plt.subplot(211)
		# plt.plot(zip(*measurementsList)[0])
		# plt.subplot(212)
		# plt.plot(zip(*measurementsList)[1])
		# plt.figure(2)
		# plt.plot(diffarray)
		# plt.figure(3)
		# plt.plot(ratarray)
		# plt.show()
		# plt.figure(1)
		# plt.plot(coverage2List)
		# plt.figure(2)
		# plt.plot(coverage2Deviance)
		# plt.show()
		# code.interact(local=locals())


if __name__ == "__main__":
	main()
