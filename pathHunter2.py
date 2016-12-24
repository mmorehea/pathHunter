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

def getCombos(l):
	combos = []
	for i, item in enumerate(list):
		n = i
		while n < len(l):
			if i == n:
				continue
			combos.append(i, n)
			n += 1
	code.interact(local=locals())

def upperLeftJustify(blob):
	box, dimensions = findBBDimensions(blob)
	transformedBlob = []
	for point in blob:
		transformedPoint = (point[0] - box[0], point[1] - box[2])
		transformedBlob.append(transformedPoint)

	return transformedBlob

def erodeAndSplit(blob, shape):
	img = np.zeros(shape, np.uint8)
	img[zip(*blob)] = 99999
	contours = []
	erodeCount = 0
	kernel = np.ones((3,3),np.uint8)

	while len(contours) < 2:

		img = cv2.erode(img.copy(),kernel,iterations = 1)
		erodeCount += 1

		contours = cv2.findContours(img.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]

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
			im = cv2.dilate(im.copy(), kernel, iterations = 1)

		subBlobs.append(zip(np.nonzero(im)[0], np.nonzero(im)[1]))

	return subBlobs

def getCandidates(regions, image):
	blobs = []
	shorthand = []
	for each in regions:
		shorthand.append(each)
		q = np.where(image==each)
		blobs.append(zip(q[0],q[1]))

	blobparents = {}
	blobchildren = []
	for blob in blobs:
		subBlobs = erodeAndSplit(blob, image.shape)
		if len(subBlobs) > 0:
			for sb in subBlobs:
				blobparents[tuple(sb)] = blob
		blobchildren.extend(subBlobs)

	blobs.extend(blobchildren)
	shorthand.extend(blobchildren)

	combosIndices = combinations(range(len(blobs)),2)
	scombosIndices = combinations(range(len(shorthand)),2)
	combos = [[blobs[i[0]], blobs[i[1]]] for i in combosIndices]
	scombos = [[shorthand[i[0]], shorthand[i[1]]] for i in scombosIndices]

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

def calculateAffinity(currentBlob, candidateBlob, normalized_xcorrelation):
	box1, dimensions1 = findBBDimensions(currentBlob)

	# Need to account for the fact that each point on the xcorrelation represents the origin of the template
	max_xcorrelation = np.max(normalized_xcorrelation[zip(*candidateBlob)])

	distance_between_centers = distance(findCentroid(currentBlob), findCentroid(candidateBlob))
	if dimensions1[0] >= dimensions1[1]:
		diameter = dimensions1[0]
	else:
		diameter = dimensions1[1]
	variance = diameter
	displacement_penalty = math.exp(-(distance_between_centers**2)/variance)

	affinity = -math.log10(max_xcorrelation * displacement_penalty)

	return affinity


# ████████ ██████   █████   ██████ ██   ██         ██████  ██████   ██████   ██████ ███████ ███████ ███████
#    ██    ██   ██ ██   ██ ██      ██  ██          ██   ██ ██   ██ ██    ██ ██      ██      ██      ██
#    ██    ██████  ███████ ██      █████           ██████  ██████  ██    ██ ██      █████   ███████ ███████
#    ██    ██   ██ ██   ██ ██      ██  ██          ██      ██   ██ ██    ██ ██      ██           ██      ██
#    ██    ██   ██ ██   ██  ██████ ██   ██ ███████ ██      ██   ██  ██████   ██████ ███████ ███████ ███████
def trackProcess(color1, maskPaths, emPaths, z, shape):
	#Initialize chain
	process = {}
	process[z] = [color1]
	maskImage1 = cv2.imread(maskPaths[z], -1)
	emImage1 = cv2.imread(emPaths[z], -1)
	currentBlob = zip(np.where(maskImage1 == color1)[0],np.where(maskImage1 == color1)[1])
	z += 1

	terminate = False
	while terminate == False:
		#FoundList is a mixed-type list that can contain any number of color IDs or lists of pixel points that, taken together, comprise the next blob in the chain
		foundList = []

		box1, dimensions1 = findBBDimensions(currentBlob)
		color1 = maskImage1[currentBlob[0]]

		if z + 1 < len(maskPaths):
			maskImage2 = cv2.imread(maskPaths[z], -1)
			emImage2 = cv2.imread(emPaths[z], -1)
		else:
			terminate = True
			continue

		emBlob = emImage1[box1[0]:box1[1],box1[2]:box1[3]]

		# Take the normalized cross correlation of the cropped currentBlob against the EM of the next slice
		normalized_xcorrelation = match_template(emImage2, emBlob)

		window = maskImage2[box1[0]:box1[1], box1[2]:box1[3]]
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

		nonzero_regions = [f[0] for f in frequency if f[0] != 0]
		# find all the candidate 2D regions and place in a list. Includes all sub-regions obtained through erosion and all combinations of regions
		candidateBlobs, candidateShorthand = getCandidates(nonzero_regions, maskImage2)

		# calculate affinity data for the candidate regions using the formula in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2630194/
		# the lower the value, the higher the affinity
		# parameters: distance between centroids, local max of normalized cross correlation
		affinityData = [calculateAffinity(currentBlob, candidate, normalized_xcorrelation) for candidate in candidateBlobs]

		code.interact(local=locals())

		q = np.where(maskImage2 == color2)
		nextBlob = zip(q[0],q[1])
		box2, dimensions2 = findBBDimensions(nextBlob)

		blob1 = upperLeftJustify(currentBlob)
		blob2 = upperLeftJustify(nextBlob)

		# figure out features that describe realtionship between shapes

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

		foundList.append(color2)
		skip = 0

		# Block out the current blob in the mask image stack
		maskImage1[zip(*currentBlob)] = 0
		cv2.imwrite(maskPaths[z], maskImage1)

		# Add foundList to the process
		process[z] = foundList

		# If not terminating, reset variables and increment z
		if terminate == False:
			currentBlob = nextBlob
			color1 = color2

			maskImage1 = maskImage2
			emImage1 = emImage2

			z += 1


	return process

# ████████ ██████   █████   ██████ ███████          ██████  ██████       ██ ███████  ██████ ████████ ███████
#    ██    ██   ██ ██   ██ ██      ██              ██    ██ ██   ██      ██ ██      ██         ██    ██
#    ██    ██████  ███████ ██      █████           ██    ██ ██████       ██ █████   ██         ██    ███████
#    ██    ██   ██ ██   ██ ██      ██              ██    ██ ██   ██ ██   ██ ██      ██         ██         ██
#    ██    ██   ██ ██   ██  ██████ ███████ ███████  ██████  ██████   █████  ███████  ██████    ██    ███████
def traceObjects(minimum_process_length, write_pickles_to, masterColorList, maskPaths, emPaths, maskShape, emShape):
	# general setup
	chainLengths = []
	objectCount = -1

	# begin searching through slices
	for z in xrange(len(maskPaths)):
		# ████████ ███████ ███████ ████████ ██ ███    ██  ██████
		#    ██    ██      ██         ██    ██ ████   ██ ██
		#    ██    █████   ███████    ██    ██ ██ ██  ██ ██   ███
		#    ██    ██           ██    ██    ██ ██  ██ ██ ██    ██
		#    ██    ███████ ███████    ██    ██ ██   ████  ██████
		if z != 656:
			continue
		#############
		# get the unique colors in that slice
		image = cv2.imread(maskPaths[z], -1)
		colorVals = [c for c in np.unique(image) if c!=0]
		###Testing###
		colorVals = [35529]
		# 24661 @1: 24762, @9:26048, @214: 26501
		# 22013 @62: 22568 @63: 21963 @67: 22517 @68: 22971, @73: 22946
		# @258: 27384
		#############

		# with all colors, begin tracing objects one by one
		for i, startColor in enumerate(colorVals):
			start = timer()
			# process is a dictionary representing a 3D process, where each key is a z index, and each value is a list of 2D regions
			process = trackProcess(startColor, maskPaths, emPaths, z, emShape)

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
				end = timer()
				print(end - start)
				print '\n'
				chainLengths.append((objectCount, color, processLength))
				pickle.dump((process, color), open(write_pickles_to + str(objectCount) + '.p', 'wb'))


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

# ██████  ██    ██ ██ ██      ██████          ██████  ███████ ███████ ██    ██ ██   ████████      ███████ ████████  █████   ██████ ██   ██
# ██   ██ ██    ██ ██ ██      ██   ██         ██   ██ ██      ██      ██    ██ ██      ██         ██         ██    ██   ██ ██      ██  ██
# ██████  ██    ██ ██ ██      ██   ██         ██████  █████   ███████ ██    ██ ██      ██         ███████    ██    ███████ ██      █████
# ██   ██ ██    ██ ██ ██      ██   ██         ██   ██ ██           ██ ██    ██ ██      ██              ██    ██    ██   ██ ██      ██  ██
# ██████   ██████  ██ ███████ ██████  ███████ ██   ██ ███████ ███████  ██████  ███████ ██ ███████ ███████    ██    ██   ██  ██████ ██   ██
def buildResultStack(write_images_to, write_pickles_to, maskPaths, maskShape):
	picklePaths = sorted(glob.glob(write_pickles_to + '*.p'))

	for z in xrange(len(maskPaths)):
		resultImg = np.zeros(maskShape, np.uint16)

		for path in picklePaths:
			process, color = pickle.load(open(path, 'rb'))

			if z in process.keys():
				for each in process[z]:
					if str(each).isdigit():
						resultImg[np.where(cv2.imread(maskPaths[z], -1)==each)] = color
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
	minimum_process_length = 0
	write_images_to = 'littleresult/'
	write_pickles_to = 'picklecrop/object'
	trace_objects = True
	build_resultStack = True
	load_stack_from_pickle_file = False
	################################################################################
	# Profiling:
	# python -m cProfile -o output pathHunter.py littlecrop/
	# python runsnake.py output

	# Get list of colors to use in the result stack
	masterColorList = pickle.load(open('masterColorList.p','rb'))

	maskFolderPath = sys.argv[1]
	emFolderPath = sys.argv[2]

	# Collect tiffs
	maskPaths =  sorted(glob.glob(maskFolderPath +'*'))
	emPaths = sorted(glob.glob(emFolderPath +'*'))

	maskShape = cv2.imread(maskPaths[0],-1).shape
	emShape = cv2.imread(emPaths[0],-1).shape

	# Make sure EM and mask data correspond
	if len(maskPaths) != len(emPaths) or maskShape != emShape:
		print 'Error, mask and EM data do not match'
		trace_objects = False
		build_resultStack = False
	else:
		# Copy all mask images into working folder so that original data is not modified
		for impath in maskPaths:
			img = cv2.imread(impath,-1)
			cv2.imwrite('workingImgStack' + impath[impath.index('/'):], img)
		maskFolderPath = 'workingImgStack/'
		maskPaths =  sorted(glob.glob(maskFolderPath +'*'))

	# Trace each process in the input stack and save as pickle file
	if trace_objects: traceObjects(minimum_process_length, write_pickles_to, masterColorList, maskPaths, emPaths, maskShape, emShape)

	# Use the pickle files to build the result stack
	if build_resultStack: buildResultStack(write_images_to, write_pickles_to, maskPaths, maskShape)

if __name__ == "__main__":
	main()

# ██     ██  ██████  ██████  ██   ██ ██ ███    ██  ██████      ██     ██ ██ ████████ ██   ██      ██████  ██████  ███    ██ ████████  ██████  ██    ██ ██████  ███████
# ██     ██ ██    ██ ██   ██ ██  ██  ██ ████   ██ ██           ██     ██ ██    ██    ██   ██     ██      ██    ██ ████   ██    ██    ██    ██ ██    ██ ██   ██ ██
# ██  █  ██ ██    ██ ██████  █████   ██ ██ ██  ██ ██   ███     ██  █  ██ ██    ██    ███████     ██      ██    ██ ██ ██  ██    ██    ██    ██ ██    ██ ██████  ███████
# ██ ███ ██ ██    ██ ██   ██ ██  ██  ██ ██  ██ ██ ██    ██     ██ ███ ██ ██    ██    ██   ██     ██      ██    ██ ██  ██ ██    ██    ██    ██ ██    ██ ██   ██      ██
#  ███ ███   ██████  ██   ██ ██   ██ ██ ██   ████  ██████       ███ ███  ██    ██    ██   ██      ██████  ██████  ██   ████    ██     ██████   ██████  ██   ██ ███████

# img8 = (img16/256).astype('uint8')
#
# contours = cv2.findContours(img8.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
# blobs = []
# for cnt in contours:
# 	mask = np.zeros(img8.shape,np.uint8)
# 	cv2.drawContours(mask,[cnt],0,255,-1)
# 	pixelpoints = np.transpose(np.nonzero(mask))
# 	blobs.append([(x[0],x[1]) for x in pixelpoints])

# convert back to row, column and store as list of points
# cnt = [(x[0][1], x[0][0]) for x in cnt]
# newconts.append(cnt)
