#!/usr/bin/env python 

import argparse 
import os
import glob
import random
import math
from caffe_tools.create_lmdb import make_lmdb


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument( "data_folder", help = "Input folder" )
  parser.add_argument( "lmdb_folder", nargs = "?", default = "", help = "LMDB folder" )
  parser.add_argument( "--keep_aspect", "-A", action = "store_true", help="Keep aspect ratio" )
  parser.add_argument( "--validation_percentage", "-V", type = float, default = 0.15, help="Percentage of dataset to use in validation" )
  parser.add_argument( "--seed", "-S", type = int, default = 0, help="Random seed value" )
  parser.add_argument( "--verbose", "-v", action="count", default = 0, help="Verbosity level" )
  args = parser.parse_args()

  data_folder = args.data_folder
  lmdb_folder = args.lmdb_folder if args.lmdb_folder != "" else "%s/lmdb" % args.data_folder
  shape = (227,227)

  validation_percentage = args.validation_percentage
  training_db_file = '%s/train_lmdb' % lmdb_folder
  validation_db_file = '%s/validation_lmdb' % lmdb_folder

  all_files = list( img for img in glob.glob( "%s/train/*jpg" % data_folder ) )
  print( "Loading %d images from %s" % ( len( all_files ), data_folder ) )
  if args.seed > 0:
    print( "Random seed value: %d" % args.seed )
    random.seed( args.seed )
  else:
    random.seed( )
  random.shuffle( all_files )
  all_data = list()
  for f in all_files:
    f_basename = os.path.basename( f )
    label = 0 if "cat" in f_basename else 1 if "dog" in f_basename else 2
    all_data.append( ( f, label ) )
  
  num_train_data = int( math.ceil( ( 1 - validation_percentage ) * len( all_data ) ) )
  training_data = all_data[0:num_train_data]
  print( "Training: writing %s (%d inputs)" % ( training_db_file, len( training_data ) ) )
  make_lmdb( training_db_file, training_data, shape, keep_aspect = args.keep_aspect, verbose = args.verbose )
  
  validation_data = all_data[num_train_data:]
  print( "Validation: writing %s (%d inputs)" % ( validation_db_file, len( validation_data ) ) )
  make_lmdb( validation_db_file, validation_data, shape, keep_aspect = args.keep_aspect, verbose = args.verbose )
  
