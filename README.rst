This is based on cats and dogs deep learning example.

Installation
------------
To install packages::

  sudo apt-get install opencl-headers build-essential protobuf-compiler libprotoc-dev libboost-all-dev libleveldb-dev hdf5-tools libhdf5-serial-dev libopencv-core-dev  libopencv-highgui-dev libsnappy-dev libsnappy1v5 libatlas-base-dev cmake libstdc++6-4.8-dbg libgoogle-glog0v5 libgoogle-glog-dev libgflags-dev liblmdb-dev git python-pip gfortran python3 ipython3 graphviz python-opencv libopencv-dev
  
  sudo apt-get install linux-image-extra-`uname -r` linux-headers-`uname -r` linux-image-`uname -r`


To install CUDA, download the network package from NVidia <https://developer.nvidia.com/cuda-downloads/>. Then::

  sudo dpkg -i cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
  sudo apt-get update
  sudo apt-get install cuda

To install cuDNN::
  
  sudo dpkg -i libcudnn5_5.1.5-1+cuda8.0_amd64.deb 
  sudo dpkg -i libcudnn5-dev_5.1.5-1+cuda8.0_amd64.deb


Download caffe:: 

  git clone https://github.com/BVLC/caffe.git

To install caffe using CMake::

  sudo mkdir /opt/caffe && sudo chown `whoami` /opt/caffe
  mkdir build && cd build 
  cmake -DCMAKE_INSTALL_PREFIX=/opt/caffe ..
  make install
  
To install python dependencies::

  sudo pip install `cat /opt/caffe/python/requirements.txt` lmdb pydot

Add the following to your .bashrc file::

  echo 'export CAFFE_ROOT=/opt/caffe' >> ~/.bashrc
  echo 'export PYTHONPATH=/opt/caffe/python:${PYTHONPATH}' >> ~/.bashrc
  echo 'export PATH=${CAFFE_ROOT}/bin:${PATH}' >> ~/.bashrc

Configuration 
-------------


