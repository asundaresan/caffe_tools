import os 
import numpy as np
import cv2
from caffe.proto import caffe_pb2
import lmdb


def transform_img( img, width, height ):
  """ Perform histogram equalization and image scaling 
  """
  img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
  img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
  img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
  img = cv2.resize(img, (width, height), interpolation = cv2.INTER_CUBIC)
  return img


def make_datum(img, label, width, height ):
  """ image is numpy.ndarray format. BGR instead of RGB
  """
  img2 = transform_img( img, width, height )
  return caffe_pb2.Datum( channels=3, width=width, height=height,
      label=label, data=np.rollaxis(img2, 2).tostring())


def make_lmdb( db_filename, data, width, height, verbose = 0 ):
  """
  """
  if not os.path.exists( db_filename ):
    print( "Creating directory: %s" % db_filename )
    os.makedirs( db_filename )
  db = lmdb.open( db_filename, map_size=int(1e12))
  with db.begin(write=True) as txn:
    for idx, (filename, label) in enumerate( data ):
      img = cv2.imread( filename, cv2.IMREAD_COLOR )
      datum = make_datum(img, label, width, height )
      txn.put('{:0>5d}'.format(idx), datum.SerializeToString())
      if verbose > 0:
        print( "%s: label=%d, shape=%s" % ( filename, label, list( img.shape ) ) )
  db.close()

