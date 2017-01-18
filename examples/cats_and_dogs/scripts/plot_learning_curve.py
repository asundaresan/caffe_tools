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


def get_plots( logfile, verbose = 0 ):
  if verbose > 0:
    print( "Loading %s" % logfile )
  train_dict_list, test_dict_list = parse_log( logfile )
  train_dict = to_dict( train_dict_list )
  test_dict = to_dict( test_dict_list )
  if verbose > 0:
    print( test_dict.keys() )
  plots = {}
  plots["Test accuracy"] = { "xval": test_dict["NumIters"], "yval": test_dict["accuracy"] }
  plots["Test loss"] = { "xval": test_dict["NumIters"], "yval": test_dict["loss"] }
  plots["Train loss"] = { "xval": train_dict["NumIters"], "yval": train_dict["loss"] }
  return plots



def plot_log( logfile, block = True, verbose = 0 ):
  handles = []
  plots = get_plots( logfile, verbose = verbose )
  for key, plot in plots.items():
    h, = plt.plot( plot["xval"], plot["yval"], label = key )
    handles.append( h )

  plt.legend( handles = handles, bbox_to_anchor=(0.01, 0.01), loc=3 )
  plt.grid( True )
  axes = plt.gca()
  axes.set_ylim( [0,1] )

  path, extension = os.path.splitext( logfile )
  outputfile = "%s.png" % path
  if verbose > 0:
    print( "Saving to file %s" % logfile )
  plt.savefig( outputfile )
  plt.show( block = block )


def compare_logs( logfile_list, block = True, verbose = 0 ):
  handles = []
  for logfile in logfile_list:
    path, extension = os.path.splitext( logfile )
    name = os.path.basename( path )
    plots = get_plots( logfile, verbose = verbose )
    for key, plot in plots.items():
      if "accuracy" in key:
        print( "%s: max accuracy = %.3f" % ( name, max( plot[ "yval" ] ) ) )
        h, = plt.plot( plot["xval"], plot["yval"], label = name )
        handles.append( h )
  plt.legend( handles = handles, bbox_to_anchor=(0.01, 0.01), loc=3 )
  plt.grid( True )
  axes = plt.gca()
  axes.set_ylim( [0,1] )

  outputfile = "compare.png"
  if verbose > 0:
    print( "Saving to file %s" % logfilename )
  plt.savefig( outputfile )
  plt.show( block = block )


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument( "log_files", nargs = "*", help = "Log filename folder" )
  parser.add_argument( "--continuous", "-C", action="store_true", help="Plot continuously" )
  parser.add_argument( "--verbose", "-v", action="count", default = 0, help="Verbosity level" )
  args = parser.parse_args()

  if args.continuous:
    while 1:
      plot_log( args.log_files, verbose = args.verbose )
      time.sleep( 1 )
  else:
    if len( args.log_files ) > 1:
      compare_logs( args.log_files )
    elif len( args.log_files ) == 1:
      plot_log( args.log_files[0] )
