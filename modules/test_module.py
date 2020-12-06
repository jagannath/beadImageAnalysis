import numpy as np
#from matplotlib import pyplot as plt
from PIL import Image
#from tqdm import tqdm
import os
import sys
import cv2
import csv
from nd2reader import ND2Reader as nd2
from statistics import mean 
from statistics import stdev
from statistics import median
#from skimage import io




s = "COmputers are uselfess"
a = [100,200]

def printy(arg):
	print("You printed {0}".format(arg))

class Classy:
	pass

class Beads(object):
	#def __init__(self,exppath,outpath,saveimgs,groupby,saveto,h1,h2)
	def __init__(self):
		self.beadname = "beads2"

	def classname(self):
		return self.beadname

