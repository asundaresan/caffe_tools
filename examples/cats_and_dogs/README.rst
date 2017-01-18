Cats and dogs example
---------------------

This example has been developed using a kaggle data set containing cats and dogs and is based on the //github.com/adilmoujahid/deeplearning-cats-dogs-tutorial.git repository. 

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
- Scale the image to a small size (227x227) (keeping aspect ratio is optional)

To create the LMDB::

  python scripts/create_lmdb.py data/images data/lmdb

If it is desired to keep the aspect ratio, use the following command instead::

  python scripts/create_lmdb.py data/images data/lmdb -A
  
Generate the mean image::

  compute_image_mean -backend=lmdb data/lmdb/train_lmdb data/mean.binaryproto

To check the image transformation used by ``create_lmdb.py``, use::

  python scripts/transform.py data/images/*.jpg -vv 

You may use the ``-A`` option to keep the aspect ratio and ``--help`` for help.

Train a cat/dog classifier
~~~~~~~~~~~~~~~~~~~~~~~~~~

Train::
  
  cd models/caffe_model_1
  caffe train --solver solver_1.prototxt 2>&1 | tee model_1_train.log

Keep track of the learning curve.::

  python ../../scripts/plot_learning_curve.py model_1_train.log -C

Train a cat/dog classifier using transfer learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Train::

  cd models/caffe_model_2
  caffe train --solver solver_2.prototxt 2>&1 | tee model_2_train.log

Keep track of the learning curve::

  python ../../scripts/plot_learning_curve.py model_2_train.log -C


