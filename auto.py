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

np.set_printoptions(threshold=np.inf)

class Blob():
	def __init__(self, imageCount, n, listofpixels, color, listofpixels_foundbelow, color_foundbelow, nFromPrevSlice, zValue, skipped):
		self.imageCount = imageCount
		self.n = n
		self.listofpixels = listofpixels
		self.color = color
		self.listofpixels_foundbelow = listofpixels_foundbelow
		self.color_foundbelow = color_foundbelow
		self.nFromPrevSlice = nFromPrevSlice
		self.zValue = zValue
		self.skipped = skipped

		self.centroid = findCentroid(listofpixels)
		self.box, self.dimensions = findBBDimensions(listofpixels)
		self.centroid_foundbelow = findCentroid(listofpixels_foundbelow)
		if len(self.listofpixels_foundbelow) > 0: self.box, self.dimensions_foundbelow = findBBDimensions(listofpixels_foundbelow)
		if len(self.listofpixels_foundbelow) > 0: self.percent_overlap_foundbelow = testOverlap(set(self.listofpixels), set(self.listofpixels_foundbelow))




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

def buildColorMap(img):
	colorMap = {0: 0}
	x, y = img.shape
	counter = 0
	uniqueValues = sorted(np.unique(img))
	for each in uniqueValues:
			if each in colorMap.values():
				continue
			else:
				counter += 1
				colorMap[counter] = each
	#print colorMap
	return colorMap

def percent_difference_in_BB_area(dim1, dim2):
	A1 = dim1[0] * dim1[1]
	A2 = dim2[0] * dim2[1]

	if A1 == 0 or A2 == 0:
		return 1
	else:
		return float(abs(A1-A2)) / A1


# /*
# ██     ██  █████  ████████ ███████ ██████  ███████ ██   ██ ███████ ██████
# ██     ██ ██   ██    ██    ██      ██   ██ ██      ██   ██ ██      ██   ██
# ██  █  ██ ███████    ██    █████   ██████  ███████ ███████ █████   ██   ██
# ██ ███ ██ ██   ██    ██    ██      ██   ██      ██ ██   ██ ██      ██   ██
#  ███ ███  ██   ██    ██    ███████ ██   ██ ███████ ██   ██ ███████ ██████
# */
def waterShed(img16):
	img8 = (img16/256).astype('uint8')
	w = np.where(img8 != 0)
	initblob = zip(w[0], w[1])

	contours = cv2.findContours(img8.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
	blobs = []
	for cnt in contours:

		mask = np.zeros(img8.shape,np.uint8)
		cv2.drawContours(mask,[cnt],0,255,-1)
		pixelpoints = np.transpose(np.nonzero(mask))
		blobs.append([(x[0],x[1]) for x in pixelpoints])

		# convert back to row, column and store as list of points
		# cnt = [(x[0][1], x[0][0]) for x in cnt]
		# newconts.append(cnt)

	if len(blobs) == 1 and blobs[0] != initblob: # causes significant speed hit
		newblob = list(set(initblob) - set(blobs[0]))
		blobs.append(newblob)


	return blobs

#  /*
# ███████ ██ ███    ██ ██████   ██████ ███████ ███    ██ ████████ ██████   ██████  ██ ██████
# ██      ██ ████   ██ ██   ██ ██      ██      ████   ██    ██    ██   ██ ██    ██ ██ ██   ██
# █████   ██ ██ ██  ██ ██   ██ ██      █████   ██ ██  ██    ██    ██████  ██    ██ ██ ██   ██
# ██      ██ ██  ██ ██ ██   ██ ██      ██      ██  ██ ██    ██    ██   ██ ██    ██ ██ ██   ██
# ██      ██ ██   ████ ██████   ██████ ███████ ██   ████    ██    ██   ██  ██████  ██ ██████
# */
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

def findSubCentroids(listofpixels, box):
	dx = box[1] - box[0]
	dy = box[3] - box[2]

	if dy > dx:
		majorAxis = 'y'
	else:
		majorAxis = 'x'

	if majorAxis == 'x':
		third = int(dx / 3)
		smallbox1 = [box[0], box[0] + third, box[2], box[3]]
		smallbox2 = [box[1] - third, box[1], box[2], box[3]]
	elif majorAxis == 'y':
		third = int(dy / 3)
		smallbox1 = [box[0], box[1], box[2], box[2] + third]
		smallbox2 = [box[0], box[1], box[3] - third, box[3]]

	smallpixels1 = [p for p in listofpixels if p[0] >= smallbox1[0] and p[0] <= smallbox1[1] and p[1] >= smallbox1[2] and p[1] <= smallbox1[3]]
	smallpixels2 = [p for p in listofpixels if p[0] >= smallbox2[0] and p[0] <= smallbox2[1] and p[1] >= smallbox2[2] and p[1] <= smallbox2[3]]

	return [findCentroid(smallpixels1), findCentroid(smallpixels2)]




# /*
# ███████ ██ ███    ██ ██████  ███    ██ ███████  █████  ██████  ███████ ███████ ████████
# ██      ██ ████   ██ ██   ██ ████   ██ ██      ██   ██ ██   ██ ██      ██         ██
# █████   ██ ██ ██  ██ ██   ██ ██ ██  ██ █████   ███████ ██████  █████   ███████    ██
# ██      ██ ██  ██ ██ ██   ██ ██  ██ ██ ██      ██   ██ ██   ██ ██           ██    ██
# ██      ██ ██   ████ ██████  ██   ████ ███████ ██   ██ ██   ██ ███████ ███████    ██
# */
def findNearest(img, startPoint):
	directions = cycle([[0,1], [1,1], [1,0], [1,-1], [0,-1], [-1,-1], [-1,0], [-1,1]])
	increment = 0
	cycleCounter = 0
	distance = [0,0]

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

#  /*
#  ██████  ███████ ████████ ███████ ███████ ███████ ██████  ██████  ██ ██   ██ ███████ ██
# ██       ██         ██    ██      ██      ██      ██   ██ ██   ██ ██  ██ ██  ██      ██
# ██   ███ █████      ██    ███████ █████   █████   ██   ██ ██████  ██   ███   █████   ██
# ██    ██ ██         ██         ██ ██      ██      ██   ██ ██      ██  ██ ██  ██      ██
#  ██████  ███████    ██    ███████ ███████ ███████ ██████  ██      ██ ██   ██ ███████ ███████
# */
def getSeedPixel(centroid, img, color):
	shouldSkip = False
	seedpixel = (0,0)


	if img[centroid] == 0:
		seedpixel = findNearest(img, centroid)
	else:
		seedpixel = centroid
	try:
		if img[seedpixel] == color:
			shouldSkip = True
			# print 'Same color! Skipping color ' + color
	except:
		shouldSkip = True
		# print 'Index out of bounds'

	return shouldSkip, seedpixel


# /*
# ████████ ███████ ███████ ████████  ██████  ██    ██ ███████ ██████  ██       █████  ██████
#    ██    ██      ██         ██    ██    ██ ██    ██ ██      ██   ██ ██      ██   ██ ██   ██
#    ██    █████   ███████    ██    ██    ██ ██    ██ █████   ██████  ██      ███████ ██████
#    ██    ██           ██    ██    ██    ██  ██  ██  ██      ██   ██ ██      ██   ██ ██
#    ██    ███████ ███████    ██     ██████    ████   ███████ ██   ██ ███████ ██   ██ ██
# */

def testOverlap(setofpixels1, setofpixels2):

	set_intersection = setofpixels1 & setofpixels2
	set_union = setofpixels1 | setofpixels2

	percent_overlap = float(len(set_intersection)) / len(set_union)

	return percent_overlap

def exclusive_or(setofpixels1, setofpixels2):
	set_intersection = setofpixels1 & setofpixels2
	set_union = setofpixels1 | setofpixels2

	return set_union - set_intersection



# /*
# ██████  ███████  ██████ ███████ ███████  █████  ██████   ██████ ██   ██
# ██   ██ ██      ██      ██      ██      ██   ██ ██   ██ ██      ██   ██
# ██████  █████   ██      ███████ █████   ███████ ██████  ██      ███████
# ██   ██ ██      ██           ██ ██      ██   ██ ██   ██ ██      ██   ██
# ██   ██ ███████  ██████ ███████ ███████ ██   ██ ██   ██  ██████ ██   ██
# */
def recSearch(pixel, img, color):
	front = [pixel]
	found = [pixel]
	foundGrid = np.zeros((img.shape[0], img.shape[1]))
	foundGrid[pixel[0], pixel[1]] = 1
	counter = 0
	while len(front) > 0:
		fronty = front
		front = []
		for each in fronty:
			pixel = eachTrue
			searchPixels = [[pixel[0]+1, pixel[1]], [pixel[0]-1, pixel[1]], [pixel[0], pixel[1]+1], [pixel[0], pixel[1]-1]]
			#code.interact(local=locals())
			for neighbor in searchPixels:
				if neighbor[0] not in range(img.shape[0]) or neighbor[1] not in range(img.shape[1]):
					#print "hit border, skipping"
					continue
				#code.interact(local=locals())
				if img[neighbor[0], neighbor[1]] == color and foundGrid[neighbor[0], neighbor[1]] == 0 and neighbor not in front:
					#code.interact(local=locals())
					front.append([neighbor[0], neighbor[1]])
					foundGrid[neighbor[0], neighbor[1]] = 1
					counter = counter + 1

					#found.append([neighbor[0], neighbor[1]])
	found = np.where(foundGrid == 1)
	found = zip(found[0],found[1])
	return found

def changeColor(img, listofpixels):
	vals = [c for c in np.unique(img) if c!=0]
	newcolor = max(vals) + 50
	if newcolor > 2**16:
		newcolor = random.choice(list(set(range(max(vals))[50:]) - set(vals)))
	for pixel in listofpixels:
		img[pixel] = newcolor
	return img



# /*
# ███    ███  █████  ██ ███    ██
# ████  ████ ██   ██ ██ ████   ██
# ██ ████ ██ ███████ ██ ██ ██  ██
# ██  ██  ██ ██   ██ ██ ██  ██ ██
# ██      ██ ██   ██ ██ ██   ████
# */

################################################################################
# SETTINGS
label_and_collect_info = False # Takes a lot more time but labels all blobs and collects info on each for use with dauto.py, good for testing.
write_images_to = 'littleresult/'
write_pickles_to = 'pickles3/blobList' # Only matters if label_and_collect_info is true
indices_of_slices_to_be_removed = []
################################################################################

dirr = sys.argv[1]

list_of_image_paths = sorted(glob.glob(dirr +'*'))

list_of_image_paths = [i for j, i, in enumerate(list_of_image_paths) if j not in indices_of_slices_to_be_removed]

images = []
for i, path in enumerate(list_of_image_paths):
	im = cv2.imread(path, -1)
	images.append(im)
	print 'Loaded image ' + str(i + 1) + '/' + str(len(list_of_image_paths))

start = timer()

zTracker = {}
chainLengths = []
# for each blob, stores 1) the n of the blob that connected to it in the previous slice and  2) a zvalue indicating how far up it is connected to other blobs
for imageCount, image1 in enumerate(images):
	print '\n'
	print imageCount
	end = timer()
	print(end - start)
	print '\n'

	colorVals = [c for c in np.unique(image1) if c!=0]

	blobList = []
	for n, color1 in enumerate(colorVals):


		where = np.where(image1 == color1)
		listofpixels1 = zip(list(where[0]), list(where[1]))

		centroid1 = findCentroid(listofpixels1)
		box, dimensions1 = findBBDimensions(listofpixels1)

		# subCentroids = findSubCentroids(listofpixels1, box)
		# finds the two centroids of the parts of the blob bounded by the two lateral thirds of the bounding box along the major axis

		if centroid1 in zTracker.keys():
			nFromPrevSlice = zTracker[centroid1][0]
			zValue = zTracker[centroid1][1]
		else:
			found = False
			imageD = np.zeros(image1.shape, np.uint16)
			for pixel in listofpixels1:
				imageD[pixel] = color1

			subBlobs = waterShed(imageD)
			branches = []
			if len(subBlobs) > 1:
				for b in subBlobs:
					c = findCentroid(b)
					if c in zTracker.keys():
						branches.append(b)
						listofpixels1 = b
						centroid1 = c
						box, dimensions1 = findBBDimensions(listofpixels1)
						nFromPrevSlice = zTracker[centroid1][0]
						zValue = zTracker[centroid1][1]
						found = True

			if found == False:
				nFromPrevSlice = None
				zValue = 0
				zTracker[centroid1] = [nFromPrevSlice, zValue]


		try:
			image2 = images[imageCount + 1]
		except:
			chainLengths.append((zValue, color1))
			del zTracker[centroid1]
			listofpixels2 = []
			color2 = 0
			shouldSkip = True
			if label_and_collect_info: blobList.append(Blob(imageCount, n, listofpixels1, color1, listofpixels2, color2, nFromPrevSlice, zValue, shouldSkip))
			continue


		shouldSkip, seedpixel = getSeedPixel(centroid1, image2, color1)

		# if percent_edgepixels(listofpixels1) > 0.5:

		if shouldSkip:
			chainLengths.append((zValue, color1))
			del zTracker[centroid1]
			listofpixels2 = []
			color2 = 0
			if label_and_collect_info: blobList.append(Blob(imageCount, n, listofpixels1, color1, listofpixels2, color2, nFromPrevSlice, zValue, shouldSkip))
			continue

		# subShouldSkip1, subseedpixel1 = getSeedPixel(subCentroids[0], image2, color1)
		# subShouldSkip2, subseedpixel2 = getSeedPixel(subCentroids[1], image2, color1)

		setofpixels1 = set(listofpixels1)

		color2 = image2[seedpixel]
		whereColor = np.where(image2==color2)
		listofpixels2 = zip(whereColor[0],whereColor[1])
		setofpixels2 = set(listofpixels2)


		percent_overlap = testOverlap(setofpixels1, setofpixels2)

		# if subShouldSkip1 == False and subShouldSkip2 == False:
		# 	smallcolors = [image2[subseedpixel1], image2[subseedpixel2]]
		#
		# 	branched = False
		# 	for smallcolor in smallcolors:
		# 		if smallcolor != color2:
		# 			w = np.where(image2==smallcolor)
		# 			smallblob = zip(w[0], w[1])
		# 			if testOverlap(setofpixels1, set(smallblob)) > 0:
		# 				biggerblob = listofpixels2 + smallblob
		# 				po = testOverlap(setofpixels1, set(biggerblob))
		# 				if po > percent_overlap:
		# 					listofpixels2 = biggerblob
		# 					percent_overlap = po
		# 					branched = True


		# listofpixels_XOR = list(exclusive_or(setofpixels1, setofpixels2))
		#
		# branch = []
		# if len(listofpixels_XOR) > 0:
		# 	Xs = •••••••••••[a[0] for a in listofpixels_XOR]
		# 	Ys = [b[1] for b in listofpixels_XOR]
		# 	c_array = image2[np.array(Xs), np.array(Ys)]
		# 	c_array = c_array[c_array != 0]
		# 	code.interact(local=locals())
		#
		# 	if len(c_array) > 0:
		# 		color3 = max(set(list(c_array)), key=list(c_array).count)
		#
		# 		wh = np.where(image2==color3)
		# 		branch = zip(wh[0], wh[1])
		#
		# new_percent_overlap = testOverlap(setofpixels1, set(listofpixels2 + branch))
		#
		# if new_percent_overlap > percent_overlap:
		# 	listofpixels2 = listofpixels2 + branch
		# 	setofpixels2 = set(listofpixels2)
		# 	percent_overlap = new_percent_overlap



		centroid2 = findCentroid(listofpixels2)
		box, dimensions2 = findBBDimensions(listofpixels2)

		# cv2.circle(image1, (seedpixel[1], seedpixel[0]), 1, int(color2), -1)
		# cv2.putText(image1, str(n), (centroid1[1],centroid1[0]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, int(color2), 1,cv2.LINE_AA)
		# cv2.line(image1, (centroid1[1],centroid1[0]), (seedpixel[1], seedpixel[0]), int(color2), 1)

		# test for branching
		# if zTracker[centroid1][1] == 0 and imageCount > 1:
		# 	image3 =  images[imageCount - 1]
		# 	# shouldSkip, seedpixel = getSeedPixel(centroid1, image3, color1)
		# 	color3 = image3[centroid1]
		# 	w =  np.where(image3==color3)
		# 	listofpixels3 = zip(w[0], w[1])
		# 	if testOverlap(setofpixels1, set(listofpixels3)) > 0:
		# 		color1 = color3
		# 		for pixel in listofpixels1:
		# 			image1[pixel] = color1



		if centroid2 in zTracker.keys():
			if zTracker[centroid1][1] < zTracker[centroid2][1]:
				del zTracker[centroid1]
				shouldSkip = True
				if label_and_collect_info: blobList.append(Blob(imageCount, n, listofpixels1, color1, listofpixels2, color2, nFromPrevSlice, zValue, shouldSkip))
				continue
		if percent_overlap == 0:
			chainLengths.append((zValue, color1))
			del zTracker[centroid1]
			shouldSkip = True
			if label_and_collect_info: blobList.append(Blob(imageCount, n, listofpixels1, color1, listofpixels2, color2, nFromPrevSlice, zValue, shouldSkip))
			continue
		elif percent_difference_in_BB_area(dimensions1, dimensions2) < 0.2: # or branched == True: #percent_overlap > 0.75 123 slices/min
			for pixel in listofpixels2:
				image2[pixel] = color1

			ww = np.where(image2==color1)
			q = zip(ww[0], ww[1])
			if q != listofpixels2:
				diff = list(set(q) - set(listofpixels2))
				image2 = changeColor(image2, diff)

			pop = zTracker.pop(centroid1)
			zTracker[centroid2] = [n, pop[1] + 1]
			if label_and_collect_info: blobList.append(Blob(imageCount, n, listofpixels1, color1, listofpixels2, color2, nFromPrevSlice, zValue, shouldSkip))
		else:
			imageD = np.zeros(image1.shape, np.uint16)
			for pixel in listofpixels2:
				imageD[pixel] = color2


			subBlobs = waterShed(imageD)
			if len(subBlobs) > 1:
				percent_overlap = 0

				for b in subBlobs:
					pt = testOverlap(setofpixels1, set(b))
					if pt > percent_overlap:
						percent_overlap = pt
						listofpixels2 = b
						setofpixels2 = set(b)
						centroid2 = findCentroid(listofpixels2)
						box, dimensions2 = findBBDimensions(listofpixels2)
				if percent_overlap == 0:
					chainLengths.append((zValue, color1))
					del zTracker[centroid1]
					shouldSkip = True
					if label_and_collect_info: blobList.append(Blob(imageCount, n, listofpixels1, color1, listofpixels2, color2, nFromPrevSlice, zValue, shouldSkip))
					continue
				else:
					for pixel in listofpixels2:
						image2[pixel] = color1

					ww = np.where(image2==color1)
					q = zip(ww[0], ww[1])
					if q != listofpixels2:
						diff = list(set(q) - set(listofpixels2))
						image2 = changeColor(image2, diff)

					pop = zTracker.pop(centroid1)
					zTracker[centroid2] = [n, pop[1] + 1]
					if label_and_collect_info: blobList.append(Blob(imageCount, n, listofpixels1, color1, listofpixels2, color2, nFromPrevSlice, zValue, shouldSkip))
			else:
				for pixel in listofpixels2:
					image2[pixel] = color1

				ww = np.where(image2==color1)
				q = zip(ww[0], ww[1])
				if q != listofpixels2:
					diff = list(set(q) - set(listofpixels2))
					image2 = changeColor(image2, diff)

				pop = zTracker.pop(centroid1)
				zTracker[centroid2] = [n, pop[1] + 1]
				if label_and_collect_info: blobList.append(Blob(imageCount, n, listofpixels1, color1, listofpixels2, color2, nFromPrevSlice, zValue, shouldSkip))


	if label_and_collect_info:
		for blob in blobList:
			cv2.putText(image1, str(blob.n), (blob.centroid[1], blob.centroid[0]), cv2.FONT_HERSHEY_COMPLEX_SMALL,  0.6, int(blob.color_foundbelow), 1,cv2.LINE_AA)
	cv2.imwrite(write_images_to + list_of_image_paths[imageCount][list_of_image_paths[imageCount].index('/')+1:], image1)
	if label_and_collect_info: pickle.dump(blobList, open(write_pickles_to + str(imageCount) + '.p', 'wb'))

print 'Number of chains: ' + str(len(chainLengths))
print 'Average chain length: ' + str(sum([x[0] for x in chainLengths])/len(chainLengths))

if os.path.exists('summary.txt'):
	os.remove('summary.txt')

chainLengths = sorted(chainLengths)[::-1]

with open('summary.txt','w') as f:
	for i,each in enumerate(chainLengths):
		f.write(str(chainLengths[i][1]) + ' ' + str(chainLengths[i][0]) + '\n')

code.interact(local=locals())
