import caffe
import numpy as np
import sys
import skimage
from skimage import transform

if len(sys.argv) != 3:
print "Usage: python convert_protomean.py proto.mean out.npy"
sys.exit()

blob = caffe.proto.caffe_pb2.BlobProto()
data = open( sys.argv[1] , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
arr = skimage.transform.resize(arr[0], (3, 227, 227))
out = arr
np.save( sys.argv[2] , out )