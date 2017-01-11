#!/usr/bin/env python 

import argparse 
import sys
import caffe
import lmdb

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument( "lmdb_folder", help = "LMDB folder" )
  parser.add_argument( "--number", "-n", type = int, default = 0, help="Number of entries to list" )
  parser.add_argument( "--verbose", "-v", action="count", default = 0, help="Verbosity level" )
  args = parser.parse_args()

  lmdb_env = lmdb.open( args.lmdb_folder )
  lmdb_txn = lmdb_env.begin()
  lmdb_cursor = lmdb_txn.cursor()
  datum = caffe.proto.caffe_pb2.Datum()

  for idx, ( key, value ) in enumerate( lmdb_cursor ):
    datum.ParseFromString(value)
    label = datum.label
    if args.number == 0 or idx < args.number:
      if args.verbose == 0:
        print( "%6d: {%s: label=%s}" % ( idx, key, label ) )
      else:
        data = caffe.io.datum_to_array(datum)
        print( "%6d: {%s: label=%s, data.shape=%s}" % ( idx, key, label, list( data.shape ) ) )
    else:
      break

