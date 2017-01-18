#!/bin/bash 
TRIALS=10
for (( c=1; c<=${TRIALS}; c++ )); do 
  echo "== Trial $c =="
  python ../../scripts/create_lmdb.py data/images data/lmdb -S ${c}
  compute_image_mean -backend=lmdb data/lmdb/train_lmdb data/mean.binaryproto
  caffe train --solver solver_1.prototxt 2>&1 | tee model_train_seed-${c}.log
  echo "== with -A option =="
  python ../../scripts/create_lmdb.py data/images data/lmdb -S ${c} -A
  compute_image_mean -backend=lmdb data/lmdb/train_lmdb data/mean.binaryproto
  caffe train --solver solver_1.prototxt 2>&1 | tee model_train_A_seed-${c}.log
done
