#!/usr/bin/env python 

import sys
import os
import time
from caffe_tools.extra.parse_log import parse_log
import yaml
import numpy as np
import matplotlib.pyplot as plt
import argparse

            
def to_dict( dict_list ):
  return { k: list( d[k] for d in dict_list ) for k in dict_list[0].keys() }


def main( logfilename, verbose = 0 ):
  if verbose > 0:
    print( "Loading %s" % logfilename )
  train_dict_list, test_dict_list = parse_log( logfilename )
  train_dict = to_dict( train_dict_list )
  test_dict = to_dict( test_dict_list )
  if verbose > 0:
    print( test_dict.keys() )
  tr_loss, = plt.plot( train_dict["NumIters"], np.array( train_dict["loss"] ), label = "Training loss" )
  te_loss, = plt.plot( test_dict["NumIters"], np.array( test_dict["loss"] ), label = "Test loss" )
  te_acc, = plt.plot( test_dict["NumIters"], test_dict["accuracy"], label = "Test accuracy" )
  plt.legend( handles = [tr_loss, te_loss, te_acc] )
  plt.show( block = False )
  path, extension = os.path.splitext( logfilename )
  pngfilename = "%s.png" % path
  if verbose > 0:
    print( "Saving to file %s" % logfilename )
  plt.savefig( pngfilename )


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument( "log_file", help = "Log filename folder" )
  parser.add_argument( "--continuous", "-C", action="store_true", help="Plot continuously" )
  parser.add_argument( "--verbose", "-v", action="count", default = 0, help="Verbosity level" )
  args = parser.parse_args()

  if args.continuous:
    while 1:
      main( args.log_file )
      time.sleep( 1 )

  main( args.log_file )
