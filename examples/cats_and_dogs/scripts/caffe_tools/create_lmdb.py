import os 
import numpy as np
import cv2
from caffe.proto import caffe_pb2
import lmdb

def resize( img, shape, keep_aspect, verbose = 0 ):
  """ Re-size the image to a fixed size
      keep_aspect: If True, the aspect ratio is retained and the rest of the image is padded with the average value
  """
  if keep_aspect == True:
    scale = min( np.array( shape, dtype = float ) / np.array( img.shape[0:2] ) )
    shape2 = tuple( ( scale * np.array( img.shape[0:2] ) ).astype( int ) )
    img2 = np.zeros( (shape[0],shape[1],3), dtype = img.dtype )
    for i in range( img2.shape[2] ):
      img2[:,:,i] = np.mean( np.mean( img[:,:,i] ) )
    img2[0:shape2[0],0:shape2[1],0:3] = cv2.resize( img, shape2[::-1], interpolation = cv2.INTER_CUBIC )
  else:
    img2 = cv2.resize(img, shape, interpolation = cv2.INTER_CUBIC)
    shape2 = shape
  if verbose > 0:
    print( "image.shape %s resized to %s (resize_shape = %s, desired=%s)" % ( 
      list( img.shape ), list( img2.shape ), list( shape2 ), list( shape ) ) )

  return img2


def transform( img, shape, keep_aspect = False, verbose = 0 ):
  """ Perform histogram equalization and image resizing 
  """
  for i in range( 3 ):
    img[:, :, i] = cv2.equalizeHist(img[:, :, i])
  img2 = resize(img, shape, keep_aspect, verbose = verbose )
  return img2


def make_datum( img, label ):
  """ image is numpy.ndarray format. BGR instead of RGB
  """
  return caffe_pb2.Datum( channels=3, width=img.shape[1], height=img.shape[0],
      label=label, data=np.rollaxis(img, 2).tostring())


def make_lmdb( db_filename, data, shape, keep_aspect = False, verbose = 0 ):
  """ Note shape = (height, width)
  """
  print( "Making LMDB: shape = %s, keep_aspect=%s" % ( list( shape ), keep_aspect ) )
  if not os.path.exists( db_filename ):
    print( "Creating directory: %s" % db_filename )
    os.makedirs( db_filename )
  db = lmdb.open( db_filename, map_size=int(1e12))
  with db.begin(write=True) as txn:
    for idx, (filename, label) in enumerate( data ):
      img = cv2.imread( filename, cv2.IMREAD_COLOR )
      img2 = transform( img, shape, keep_aspect = True )
      datum = make_datum(img2, label )
      txn.put('{:0>5d}'.format(idx), datum.SerializeToString())
      if verbose > 0:
        print( "%s: label=%d, shape=%s" % ( filename, label, list( img.shape ) ) )
  db.close()

