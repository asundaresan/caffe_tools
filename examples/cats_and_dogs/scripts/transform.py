#!/usr/bin/env python 

import cv2
import os
import numpy as np
import argparse
from caffe_tools.create_lmdb import transform


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument( "image_files", nargs = "*", default = "", help = "Image files to process" )
  parser.add_argument( "--keep_aspect", "-A", action = "store_true", help="Keep aspect ratio" )
  parser.add_argument( "--verbose", "-v", action="count", default = 0, help="Verbosity level" )
  args = parser.parse_args()

  shape = (200,200)
  for image_file in args.image_files:
    img = cv2.imread( image_file, cv2.IMREAD_COLOR )
    img2 = transform( img, shape, keep_aspect = args.keep_aspect, verbose = args.verbose )
    if args.verbose > 1:
      cv2.imshow( "Original", img )
      cv2.imshow( "Processed", img2 )
      ret = cv2.waitKey()
      if chr( ret ).lower() == "q":
        break


