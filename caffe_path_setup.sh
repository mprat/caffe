#!/bin/bash

# set up libraries for caffe
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/vision/torralba/datasetbias/caffe/glog-0.3.3/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/vision/torralba/sentences/glog/lib
#export LD_LIBRARY_PATH=/data/vision/torralba/datasetbias/caffe/cuda-5.5/lib64:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/data/vision/torralba/datasetbias/caffe-cudnn/cuda-7.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data/vision/torralba/sentences/lib/cuda-7.0/lib64:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/data/vision/torralba/aditya_datasets/intel/mkl/lib/intel64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data/vision/torralba/sentences/lib/intel/mkl/lib/intel64:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/data/vision/torralba/sentences/lib/protobuf/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/data/vision/torralba/sentences/lib/protobuf2/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/data/vision/torralba/sentences/lib/protobuf3/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/data/vision/torralba/datasetbias/caffe/protobuf2/protobuf-2.4.1/lib$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/data/vision/torralba/datasetbias/caffe/protobuf-2.5.0/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/data/vision/torralba/mooc-video/boost_build_1_55/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data/vision/torralba/sentences/lib/boost/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data/vision/torralba/sentences/lib/gflags/build/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/data/vision/torralba/sentences/lib/cudnn:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data/vision/torralba/sentences/lib/cudnn-rc2:$LD_LIBRARY_PATH
export KMP_DUPLICATE_LIB_OK=TRUE

# set up path for youtube-dl and ffmpeg
export PATH=/afs/csail.mit.edu/u/m/mprat/bin:$PATH
