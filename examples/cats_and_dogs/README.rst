Cats and dogs example
---------------------

This example has been developed using a kaggle data set containing cats and dogs. 

All the instructions are to be performed from the cats_and_dogs examples folder.

Download data
~~~~~~~~~~~~~
Download the training and test data into the following folders. ::

  cd data 
  unzip test.zip 
  unzip train1.zip

Pre-processing the data 
~~~~~~~~~~~~~~~~~~~~~~~
The first step processes the input images to create the LMDB set for training and validation. It does the following 
- Perform histogram equalization across the channels
- Scale the image to a small size (227x227)
  Does this mean that the aspect ratio is changed?

Run::

  python scripts/create_lmdb.py data/images data/lmdb
  
Generate the mean image::

  compute_image_mean -backend=lmdb data/lmdb/train_lmdb data/mean.binaryproto

Train::
  
  cd models/caffe_model_1
  caffe train --solver solver_1.prototxt 2>&1 | tee model_1_train.log

Keep track of the learning curve.::

  python ~/projects/deeplearning-cats-dogs-tutorial/caffe_models/code/plot_learning_curve.py ~/cats_and_dogs/model_1_train.log ~/cats_and_dogs/caffe_model_1_learning_curve.png

