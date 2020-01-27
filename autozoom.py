#!/usr/bin/env python

import torch
import torchvision

import base64
import cupy
import cv2
import flask
import argparse
import gevent
import gevent.pywsgi
import h5py
import io
import math
import moviepy
import moviepy.editor
import numpy
import os
import random
import re
import scipy
import scipy.io
import shutil
import sys
import tempfile
import time
import urllib
import zipfile

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:3])) >= 120) # requires at least pytorch version 1.2.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

objectCommon = {}

exec(open('./common.py', 'r').read())

exec(open('./models/disparity-estimation.py', 'r').read())
exec(open('./models/disparity-adjustment.py', 'r').read())
exec(open('./models/disparity-refinement.py', 'r').read())
exec(open('./models/pointcloud-inpainting.py', 'r').read())

##########################################################


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--in", required=True,
	help="path to the input image", type="str", default="./images/doublestrike.jpg")
ap.add_argument("--out", required=True,
	help="path to where the output should be stored", type="str", default="./autozoom.mp4")
ap.add_argument("--dblShift", required=False,
	help="dblShift", type="float", default=100.0)
ap.add_argument("--dblZoom", required=False,
	help="dblZoom", type="float", default=1.25)
ap.add_argument("--dblSteps", required=False,
	help="dblSteps", type="str", default="0.0 1.0 75"


args = vars(ap.parse_args())
# end

##########################################################

if __name__ == '__main__':
	numpyImage = cv2.imread(filename=arguments_strIn, flags=cv2.IMREAD_COLOR)

	intWidth = numpyImage.shape[1]
	intHeight = numpyImage.shape[0]

	dblRatio = float(intWidth) / float(intHeight)

	intWidth = min(int(1024 * dblRatio), 1024)
	intHeight = min(int(1024 / dblRatio), 1024)

	numpyImage = cv2.resize(src=numpyImage, dsize=(intWidth, intHeight), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)

	process_load(numpyImage, {})

	objectFrom = {
		'dblCenterU': intWidth / 2.0,
		'dblCenterV': intHeight / 2.0,
		'intCropWidth': int(math.floor(0.97 * intWidth)),
		'intCropHeight': int(math.floor(0.97 * intHeight))
	}

	objectTo = process_autozoom({
		'dblShift': args['dblShift'],
		'dblZoom': args['dblZoom'],
		'objectFrom': objectFrom
	})

	numpyResult = process_kenburns({
		'dblSteps': numpy.linspace(args['dblSteps'].split()[0], args['dblSteps'].split()[1], args['dblSteps'].split()[2]).tolist(),
		'objectFrom': objectFrom,
		'objectTo': objectTo,
		'boolInpaint': True
	})

	moviepy.editor.ImageSequenceClip(sequence=[ numpyFrame[:, :, ::-1] for numpyFrame in numpyResult + list(reversed(numpyResult))[1:] ], fps=25).write_videofile(arguments_strOut)
# end
