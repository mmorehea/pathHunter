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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import data
from skimage.feature import match_template
import math

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
def testOverlap(setofpixels1, setofpixels2):
	set_intersection = setofpixels1 & setofpixels2
	set_union = setofpixels1 | setofpixels2
	percent_overlap = float(len(set_intersection)) / len(set_union)

	return percent_overlap
def distance(point1, point2):
	return ((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)**0.5
def transformBlob(blob, displacement):
	dx, dy = displacement

	transformedBlob = []
	for point in blob:
		newPoint = (point[0] + dx, point[1] + dy)
		transformedBlob.append(newPoint)

	return transformedBlob
def shapeMatch(blob1, blob2, shape):
	box1, dimensions1 = findBBDimensions(blob1)
	box2, dimensions2 = findBBDimensions(blob2)
	if 0 in dimensions1 or 0 in dimensions2:
		return sys.maxint

	img1 = np.zeros(shape, np.uint8)
	img2 = img1.copy()
	img1[zip(*blob1)] = 99999
	img2[zip(*blob2)] = 99999
	try:
		im, contours, hierarchy = cv2.findContours(img1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		cnt1 = contours[0]
		im, contours, hierarchy = cv2.findContours(img2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		cnt2 = contours[0]
	except:
		return sys.maxint

	match = cv2.matchShapes(cnt1, cnt2, 1, 0)
	return match
def verifyShorthand(nextBlob, nextBlobShorthand, maskImage2):
	testBlob = []
	for each in nextBlobShorthand:
		if each == None:
			continue
		if str(each).isdigit():
			a = np.where(maskImage2==each)
			testBlob.extend(zip(a[0],a[1]))
		else:
			testBlob.extend(each)
	return nextBlob == testBlob
def dilated(blob, shape, iterations):
	img = np.zeros(shape,np.uint16)
	img[zip(*blob)] = 99999
	kernel = np.ones((3,3),np.uint8)
	dilated = cv2.dilate(img, kernel, iterations=iterations)
	nonz = np.nonzero(dilated)
	dilatedBlob = zip(nonz[0],nonz[1])

	return dilatedBlob
def erodeAndSplit(blob, shape, currentBlob, z):
	box1, dimensions1 = findBBDimensions(currentBlob)
	centroid1 = findCentroid(currentBlob)
	splitRange = np.max(dimensions1)

	img = np.zeros(shape, np.uint8)
	img[zip(*blob)] = 99999
	contours = []
	erodeCount = 0
	kernel = np.ones((3,3),np.uint8)
	haveSplit = False

	while len(contours) < 2:
		img = cv2.erode(img,kernel,iterations = 1)
		erodeCount += 1

		contours = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
		if len(contours) > 1: haveSplit = True

		indices_too_far_away = []
		for i, each in enumerate(contours):
			M = cv2.moments(each)
			try:
				contcenter = (int(M['m01']/M['m00']), int(M['m10']/M['m00']))
				if distance(centroid1, contcenter) > splitRange:
					indices_too_far_away.append(i)
			except ZeroDivisionError:
				indices_too_far_away.append(i)

		contours = [cnt for n, cnt in enumerate(contours) if n not in indices_too_far_away]

		if len(contours) == 1 and haveSplit == True:
			M = cv2.moments(contours[0])
			try:
				contcenter = (int(M['m01']/M['m00']), int(M['m10']/M['m00']))
				if distance(centroid1, contcenter) < 0.2 * splitRange:
					break
			except ZeroDivisionError:
				pass

		if erodeCount > 4:
			return []

	subBlobs = []
	for cnt in contours:
		mask = np.zeros(img.shape,np.uint8)
		cv2.drawContours(mask,[cnt],0,255,-1)
		pixelpoints = np.transpose(np.nonzero(mask))
		b = [(x[0],x[1]) for x in pixelpoints]

		im = np.zeros(shape,np.uint16)
		im[zip(*b)] = 99999
		for c in xrange(erodeCount):
			dim = cv2.dilate(im, kernel, iterations = 1)
		nz = np.nonzero(dim)
		sb = zip(nz[0], nz[1])

		subBlobs.append(sb)

	return subBlobs
def getCandidates(regions, image, currentBlob, z):
	blobs = []
	shorthand = []
	for each in regions:
		shorthand.append((each, None))
		q = np.where(image==each)
		blobs.append(zip(q[0],q[1]))
	blobparents = {}
	blobchildren = []
	for blob in blobs:
		subBlobs = erodeAndSplit(blob, image.shape, currentBlob, z)
		if len(subBlobs) > 0:
			for sb in subBlobs:
				blobparents[tuple(sb)] = blob
			blobchildren.extend(subBlobs)

	blobs.extend(blobchildren)
	shorthand.extend([(each, None) for each in blobchildren])

	combosIndices = combinations(range(len(blobs)),2)
	scombosIndices = combinations(range(len(shorthand)),2)
	combos = [(blobs[i[0]], blobs[i[1]]) for i in combosIndices]
	scombos = [(shorthand[i[0]][0], shorthand[i[1]][0]) for i in scombosIndices]

	indicesToRemove = []

	# Rules:
	# 1. no combo can include a parent and any of its children
	# 2. no combo can include a whole set of siblings
	for i, combo in enumerate(combos):
		if tuple(combo[0]) in blobparents.keys():
			if blobparents[tuple(combo[0])] == combo[1]:
				indicesToRemove.append(i)
		if tuple(combo[1]) in blobparents.keys():
			if blobparents[tuple(combo[1])] == combo[0]:
				indicesToRemove.append(i)
		if tuple(combo[0]) in blobparents.keys() and tuple(combo[1]) in blobparents.keys():
			if blobparents[tuple(combo[0])] == blobparents[tuple(combo[1])]:
				indicesToRemove.append(i)

	combos = [combo for i, combo in enumerate(combos) if i not in indicesToRemove]
	scombos = [scombo for i, scombo in enumerate(scombos) if i not in indicesToRemove]

	for combo in combos: blobs.append(combo[0] + combo[1])
	shorthand.extend(scombos)

	if len(shorthand) != len(blobs):
		print 'Something went wrong, shorthand is different from blobs'
		code.interact(local=locals())

	return blobs, shorthand
def calculateAffinity(currentBlob, candidateBlob, emImage1, emImage2, z):

	box1, dimensions1 = findBBDimensions(currentBlob)
	emBlob = np.zeros(emImage1.shape, np.uint8)
	emBlob[zip(*currentBlob)] = emImage1[zip(*currentBlob)]
	for i, each in enumerate(box1):
		if i % 2 == 0:
			if each == 0:
				box1[i] += 1
		elif i == 1:
			if each == emImage1.shape[0]:
				box1[i] -= 1
		elif i == 3:
			if each == emImage1.shape[1]:
				box1[i] -= 1

	emBlob = emBlob[box1[0]-1:box1[1]+1,box1[2]-1:box1[3]+1]

	box2, dimensions2 = findBBDimensions(candidateBlob)
	if emBlob.size < dimensions2[0] * dimensions2[1]:
		emBlob = cv2.resize(emBlob, (dimensions2[1],dimensions2[0]))

	image2 = np.zeros(emImage2.shape,np.uint8)
	image2[zip(*candidateBlob)] = emImage2[zip(*candidateBlob)]

	normalized_xcorrelation = match_template(image2, emBlob)

	try:
		max_xcorrelation = np.max(normalized_xcorrelation)
	except:
		affinity = sys.maxint


	distance_between_centers = distance(findCentroid(currentBlob), findCentroid(candidateBlob))
	diameter = np.max(dimensions1)
	variance = diameter
	displacement_penalty = math.exp(-(distance_between_centers**2)/variance)

	if max_xcorrelation <= 0.00000000001 or displacement_penalty <= 0.00000000001:
		affinity = sys.maxint
	else:
		try:
			affinity = -math.log10(max_xcorrelation * displacement_penalty)
		except: affinity = sys.maxint

	return affinity
def trackProcess(color1, maskImages, emImages, z, shape):
	#Initialize chain
	process = {}
	process[z] = (color1, None)
	maskImage1 = maskImages[z]
	emImage1 = emImages[z]
	whereCurrentBlob = np.where(maskImage1 == color1)
	currentBlob = zip(whereCurrentBlob[0], whereCurrentBlob[1])
	skipcount = 0
	d = 0
	z += 1

	terminate = False

	while terminate == False:
		if z % 50 == 0 or z == len(maskImages) - 1:
			print '\t' + str(z)

		box1, dimensions1 = findBBDimensions(currentBlob)

		# So that the normalized_xcorrelation doesn't give an error due to the template being too small
		if dimensions1[0] == 0 or dimensions1[1] == 0:
			terminate = True
			continue

		# If you have had to skip more than 60 slices, terminate
		if skipcount > 60:
			terminate = True
			continue

		# When you have reached the end of the stack, terminate
		if z + 1 < len(maskImages):
			maskImage2 = maskImages[z]
			emImage2 = emImages[z]
		else:
			terminate = True
			continue

		organicWindow = maskImage2[zip(*currentBlob)]
		frequency = collections.Counter(organicWindow).most_common()

		# Check for blackness. If you find mostly nothing 12 slices in a row, terminate
		if frequency[0][0] == 0 and len(frequency) == 1:
			if d > 12:
				terminate = True
				continue
			else:
				d += 1
				z += 1
				continue

		# Find all regions overlapping the current blob by more than 15 %
		nonzero_regions = [f[0] for f in frequency if f[0] != 0 and float(f[1])/len(currentBlob) > 0.15]

		# find all the candidate 2D regions and place in a list. Includes all sub-regions obtained through erosion and all combinations of regions
		candidateBlobs, candidateShorthand = getCandidates(nonzero_regions, maskImage2, currentBlob, z)

		if len(candidateBlobs) > 10:
			skipcount += 1
			z += 1
			continue

		# calculate affinity data for the candidate regions using the formula in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2630194/
		# the lower the value, the higher the affinity
		# parameters: distance between centroids, local max of normalized cross correlation
		affinityData = [calculateAffinity(dilated(currentBlob,shape,2), dilated(candidate,shape,2), emImage1, emImage2, z) for candidate in candidateBlobs]

		if len(affinityData) == 0:
			skipcount += 1
			z += 1
			continue

		if np.min(affinityData) > 100:
			skipcount += 1
			z += 1
			continue

		i = np.where(affinityData==np.min(affinityData))[0][0]
		nextBlob = candidateBlobs[i]
		nextBlobShorthand = candidateShorthand[i]

		# if verifyShorthand(nextBlob, nextBlobShorthand, maskImage2) == False:
		# 	print 'ERROR: nextBlob and nextBlobShorthand do not match.'
		# 	code.interact(local=locals())

		freq2 = len(set(currentBlob) & set(nextBlob))
		coverage2 = freq2 / float(len(nextBlob))

		# If the shape changes dramatically, terminate
		if coverage2 < 0.7 and shapeMatch(currentBlob,nextBlob,shape) > 0.35:
			skipcount += 1
			z += 1
			continue

		# Add shorthand for the next blob to the process
		process[z] = nextBlobShorthand

		# If not terminating, reset variables, block out the current blob and increment z
		if terminate == False:
			d = 0
			skipcount = 0

			maskImage1[zip(*currentBlob)] = 0

			currentBlob = nextBlob
			maskImage1 = maskImage2
			emImage1 = emImage2

			z += 1

	maskImage1[zip(*currentBlob)] = 0
	return process, maskImages
def traceObjects(start, minimum_process_length, write_pickles_to, masterColorList, maskImages, emImages, maskShape, emShape):
	# general setup
	chainLengths = []
	objectCount = -1

	# Search through slices to get all chains that start more than 500 slices above the end of the stack
	for z in xrange(len(maskImages[:-500])):
		###TESTING###
		if z != 0:
			continue
		#############

		# get the unique colors in that slice, order by size of blob
		image = maskImages[z]
		colorVals = [c for c in np.unique(image) if c!=0]
		blobs = []
		c = {}
		for color in colorVals:
			wblob = np.where(image==color)
			blob = zip(wblob[0], wblob[1])
			blobs.append(blob)
			c[tuple(blob)] = color
		blobs = sorted(blobs, key=len)
		colorVals = [c[tuple(blob)] for blob in blobs]
		###Testing###
		# Very tricky: 21559
		colorVals = [22013, 25140, 24081, 23324, 19063]
		#############

		# with all colors, begin tracing objects one by one
		for i, startColor in enumerate(colorVals):
			# process is a dictionary representing a 3D process, where each key is a z index, and each value is a tuple of 2D regions
			process, maskImages = trackProcess(startColor, maskImages, emImages, z, emShape)

			processLength = np.max(np.array(process.keys())) - np.min(np.array(process.keys()))

			if processLength > minimum_process_length:
				objectCount += 1

				# This block ensures that each new process gets assigned a unique color in the 16 bit range
				if objectCount < len(masterColorList):
					color = masterColorList[objectCount]
				else:
					while True:
						color = random.choice(range(2**16))
						if color != 0 and color not in masterColorList:
							masterColorList.append(color)
							break
						if len(masterColorList) >= 2**16 - 1:
							print 'ERROR: Too many objects for color range.'

				print '\n'
				print objectCount
				print 'Started at z=' + str(z)
				end = timer()
				print(end - start)
				print '\n'
				chainLengths.append((objectCount, color, processLength))
				pickle.dump((process, color), open(write_pickles_to + str(objectCount) + '.p', 'wb'))
				pickle.dump(chainLengths, open('chainLengths.p','wb'))
def summarize():
	chainLengths = pickle.load(open('chainLengths.p','rb'))
	print 'Number of chains: ' + str(len(chainLengths))
	print 'Average chain length: ' + str(float(sum([x[2] for x in chainLengths]))/len(chainLengths))
	print '\nSummarizing...'
	# print s

	if os.path.exists('summary.txt'):
		os.remove('summary.txt')

	# Need to make sure this does what I expect it to:
	chainLengths = sorted(chainLengths)[::-1]

	# Summarize information on the chains that were found
	with open('summary.txt','w') as f:
		for i,each in enumerate(chainLengths):
			f.write(str(chainLengths[i][0]) + ' ' + str(chainLengths[i][1]) + ' ' + str(chainLengths[i][2]) + '\n')
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

		cv2.imwrite(write_images_to + maskPaths[z][maskPaths[z].index('/')+1:], resultImg)
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
	write_images_to = 'littleresult/'
	write_pickles_to = 'picklecrop/object'
	trace_objects = True
	summarize_chains = True
	build_resultStack = True
	################################################################################
	# Profiling:
	# python -m cProfile -o output pathHunter.py littlecrop/
	# python runsnake.py output

	# Get list of colors to use in the result stack
	masterColorList = pickle.load(open('masterColorList.p','rb'))

	maskFolderPath = sys.argv[1]
	emFolderPath = sys.argv[2]
	maskPaths =  sorted(glob.glob(maskFolderPath +'*'))
	emPaths = sorted(glob.glob(emFolderPath +'*'))
	maskImages = [cv2.imread(maskPaths[z], -1) for z in xrange(len(maskPaths))]
	emImages = [cv2.imread(emPaths[z], -1) for z in xrange(len(emPaths))]

	maskShape = maskImages[0].shape
	emShape = emImages[0].shape

	# Make sure EM and mask data correspond
	if len(maskPaths) != len(emPaths) or maskShape != emShape:
		print 'Error, mask and EM data do not match'
		trace_objects = False
		build_resultStack = False

	startTime = timer()

	# Trace each process in the input stack and save as pickle file
	if trace_objects: traceObjects(startTime, minimum_process_length, write_pickles_to, masterColorList, maskImages, emImages, maskShape, emShape)

	# Summarize the chains in the result stack, their lengths and colors
	if summarize_chains: summarize()

	# Use the pickle files to build the result stack
	if build_resultStack: buildResultStack(startTime, write_images_to, write_pickles_to, maskPaths, maskShape)

if __name__ == "__main__":
	main()
