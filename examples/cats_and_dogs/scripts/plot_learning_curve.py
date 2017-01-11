#!/usr/bin/env python 

import sys
from extra.parse_log import parse_log
import yaml
import numpy as np
import matplotlib.pyplot as plt

            
def to_structured_array( dict_list ):
  names = dict_list[0].keys()
  dtype = [(a,"f8") for a in dict_list[0].keys() ]
  array = np.array( list(d.values() for d in dict_list), dtype = dtype )
  return array

def to_dict( dict_list ):
  return { k: list( d[k] for d in dict_list ) for k in dict_list[0].keys() }


def main( logfilename):
  print( "Loading %s" % logfilename )
  train_dict_list, test_dict_list = parse_log( logfilename )
  train_dict = to_dict( train_dict_list )
  test_dict = to_dict( test_dict_list )
  print( test_dict.keys() )
  tr_loss, =plt.plot( train_dict["NumIters"], np.array( train_dict["loss"] )*1e-2, label = "Training loss" )
  te_loss, =plt.plot( test_dict["NumIters"], np.array( test_dict["loss"] )*1e-2, label = "Test loss" )
  te_acc, =plt.plot( test_dict["NumIters"], test_dict["accuracy"], label = "Test accuracy" )
  plt.legend( handles = [tr_loss, te_loss, te_acc] )
  plt.show()


if __name__ == "__main__":
  main( sys.argv[1] )
