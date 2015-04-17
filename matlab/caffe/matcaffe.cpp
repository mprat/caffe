//
// matcaffe.cpp provides a wrapper of the caffe::Net class as well as some
// caffe::Caffe functions so that one could easily call it from matlab.
// Note that for matlab, we will simply use float as the data type.

#include <sstream>
#include <string>
#include <vector>

#include "mex.h"

#include "caffe/caffe.hpp"

#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs

#define CHECK_EQ_X(A, B, C, FUNC, ...) FUNC

// The macro that the programmer uses
#define CHECK_EQ_MEX(...)  CHECK_EQ_X(__VA_ARGS__,  \
                       CHECK_EQ_3(__VA_ARGS__),    \
                       CHECK_EQ_2(__VA_ARGS__))

#define CHECK_EQ_2(a, b)  do {                                  \
  if ( (a) != (b) ) {                                           \
    fprintf(stderr, "%s:%d: Check failed because %s != %s\n",   \
            __FILE__, __LINE__, #a, #b);                        \
    mexErrMsgTxt("Error in CHECK");                             \
  }                                                             \
} while (0);

#define CHECK_EQ_3(a, b, m)  do {                               \
  if ( (a) != (b) ) {                                           \
    fprintf(stderr, "%s:%d: Check failed because %s != %s\n",   \
            __FILE__, __LINE__, #a, #b);                        \
    fprintf(stderr, "%s:%d: %s\n",                              \
            __FILE__, __LINE__, #m);                            \
    mexErrMsgTxt(#m);                                           \
  }                                                             \
} while (0);

#define CUDA_CHECK_MEX(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ_MEX(error, cudaSuccess, "CUDA_CHECK")  \
  } while (0)
  
// Log and throw a Mex error
inline void mex_error(const std::string &msg) {
  LOG(ERROR) << msg;
  mexErrMsgTxt(msg.c_str());
}

using namespace caffe;  // NOLINT(build/namespaces)

// The pointer to the internal caffe::Net instance
static shared_ptr<Net<float> > net_;
static int init_key = -2;

static void CheckFile(const string& filename) {
    std::ifstream f(filename.c_str());
    CHECK_EQ_MEX(f.good(), true, "Could not open file " + filename);
    f.close();
}

// Five things to be aware of:
//   caffe uses row-major order
//   matlab uses column-major order
//   caffe uses BGR color channel order
//   matlab uses RGB color channel order
//   images need to have the data mean subtracted
//
// Data coming in from matlab needs to be in the order
//   [width, height, channels, images]
// where width is the fastest dimension.
// Here is the rough matlab for putting image data into the correct
// format:
//   % convert from uint8 to single
//   im = single(im);
//   % reshape to a fixed size (e.g., 227x227)
//   im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
//   % permute from RGB to BGR and subtract the data mean (already in BGR)
//   im = im(:,:,[3 2 1]) - data_mean;
//   % flip width and height to make width the fastest dimension
//   im = permute(im, [2 1 3]);
//
// If you have multiple images, cat them with cat(4, ...)
//
// The actual forward function. It takes in a cell array of 4-D arrays as
// input and outputs a cell array.

static mxArray* do_forward(const mxArray* const bottom) {
  const vector<Blob<float>*>& input_blobs = net_->input_blobs();
  if (static_cast<unsigned int>(mxGetDimensions(bottom)[0]) !=
      input_blobs.size()) {
    mex_error("Invalid input size");
  }
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(bottom, i);
    if (!mxIsSingle(elem)) {
      mex_error("MatCaffe require single-precision float point data");
    }
    if (mxGetNumberOfElements(elem) != input_blobs[i]->count()) {
      std::string error_msg;
      error_msg += "MatCaffe input size does not match the input size ";
      error_msg += "of the network";
      mex_error(error_msg);
    }

    const float* const data_ptr =
        reinterpret_cast<const float* const>(mxGetPr(elem));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(input_blobs[i]->count(), data_ptr,
          input_blobs[i]->mutable_cpu_data());
      break;
    case Caffe::GPU:
      caffe_copy(input_blobs[i]->count(), data_ptr,
          input_blobs[i]->mutable_gpu_data());
      break;
    default:
      mex_error("Unknown Caffe mode");
    }  // switch (Caffe::mode())
  }
  const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled();
  mxArray* mx_out = mxCreateCellMatrix(output_blobs.size(), 1);
  for (unsigned int i = 0; i < output_blobs.size(); ++i) {
    // internally data is stored as (width, height, channels, num)
    // where width is the fastest dimension
    mwSize dims[4] = {output_blobs[i]->width(), output_blobs[i]->height(),
      output_blobs[i]->channels(), output_blobs[i]->num()};
    mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    mxSetCell(mx_out, i, mx_blob);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(output_blobs[i]->count(), output_blobs[i]->cpu_data(),
          data_ptr);
      break;
    case Caffe::GPU:
      caffe_copy(output_blobs[i]->count(), output_blobs[i]->gpu_data(),
          data_ptr);
      break;
    default:
      mex_error("Unknown Caffe mode");
    }  // switch (Caffe::mode())
  }

  return mx_out;
}

static mxArray* do_backward(const mxArray* const top_diff) {
  const vector<Blob<float>*>& output_blobs = net_->output_blobs();
  const vector<Blob<float>*>& input_blobs = net_->input_blobs();
  if (static_cast<unsigned int>(mxGetDimensions(top_diff)[0]) !=
      output_blobs.size()) {
    mex_error("Invalid input size");
  }
  // First, copy the output diff
  for (unsigned int i = 0; i < output_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(top_diff, i);
    const float* const data_ptr =
        reinterpret_cast<const float* const>(mxGetPr(elem));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(output_blobs[i]->count(), data_ptr,
          output_blobs[i]->mutable_cpu_diff());
      break;
    case Caffe::GPU:
      caffe_copy(output_blobs[i]->count(), data_ptr,
          output_blobs[i]->mutable_gpu_diff());
      break;
    default:
        mex_error("Unknown Caffe mode");
    }  // switch (Caffe::mode())
  }
  // LOG(INFO) << "Start";
  net_->Backward();
  // LOG(INFO) << "End";
  mxArray* mx_out = mxCreateCellMatrix(input_blobs.size(), 1);
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    // internally data is stored as (width, height, channels, num)
    // where width is the fastest dimension
    mwSize dims[4] = {input_blobs[i]->width(), input_blobs[i]->height(),
      input_blobs[i]->channels(), input_blobs[i]->num()};
    mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    mxSetCell(mx_out, i, mx_blob);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(input_blobs[i]->count(), input_blobs[i]->cpu_diff(), data_ptr);
      break;
    case Caffe::GPU:
      caffe_copy(input_blobs[i]->count(), input_blobs[i]->gpu_diff(), data_ptr);
      break;
    default:
        mex_error("Unknown Caffe mode");
    }  // switch (Caffe::mode())
  }

  return mx_out;
}

static mxArray* do_get_weights() {
  const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
  const vector<string>& layer_names = net_->layer_names();

  // Step 1: count the number of layers with weights
  int num_layers = 0;
  {
    string prev_layer_name = "";
    for (unsigned int i = 0; i < layers.size(); ++i) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }
      if (layer_names[i] != prev_layer_name) {
        prev_layer_name = layer_names[i];
        num_layers++;
      }
    }
  }

  // Step 2: prepare output array of structures
  mxArray* mx_layers;
  {
    const mwSize dims[2] = {num_layers, 1};
    const char* fnames[2] = {"weights", "layer_names"};
    mx_layers = mxCreateStructArray(2, dims, 2, fnames);
  }

  // Step 3: copy weights into output
  {
    string prev_layer_name = "";
    int mx_layer_index = 0;
    for (unsigned int i = 0; i < layers.size(); ++i) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }

      mxArray* mx_layer_cells = NULL;
      if (layer_names[i] != prev_layer_name) {
        prev_layer_name = layer_names[i];
        const mwSize dims[2] = {static_cast<mwSize>(layer_blobs.size()), 1};
        mx_layer_cells = mxCreateCellArray(2, dims);
        mxSetField(mx_layers, mx_layer_index, "weights", mx_layer_cells);
        mxSetField(mx_layers, mx_layer_index, "layer_names",
            mxCreateString(layer_names[i].c_str()));
        mx_layer_index++;
      }

      for (unsigned int j = 0; j < layer_blobs.size(); ++j) {
        // internally data is stored as (width, height, channels, num)
        // where width is the fastest dimension
        mwSize dims[4] = {layer_blobs[j]->width(), layer_blobs[j]->height(),
            layer_blobs[j]->channels(), layer_blobs[j]->num()};

        mxArray* mx_weights =
          mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
        mxSetCell(mx_layer_cells, j, mx_weights);
        float* weights_ptr = reinterpret_cast<float*>(mxGetPr(mx_weights));

        switch (Caffe::mode()) {
        case Caffe::CPU:
          caffe_copy(layer_blobs[j]->count(), layer_blobs[j]->cpu_data(),
              weights_ptr);
          break;
        case Caffe::GPU:
          caffe_copy(layer_blobs[j]->count(), layer_blobs[j]->gpu_data(),
              weights_ptr);
          break;
        default:
          mex_error("Unknown Caffe mode");
        }
      }
    }
  }

  return mx_layers;
}

static void get_weights(MEX_ARGS) {
  plhs[0] = do_get_weights();
}

static void set_mode_cpu(MEX_ARGS) {
  Caffe::set_mode(Caffe::CPU);
}

static void set_mode_gpu(MEX_ARGS) {
  Caffe::set_mode(Caffe::GPU);
}

static void set_device(MEX_ARGS) {
  if (nrhs != 1) {
    ostringstream error_msg;
    error_msg << "Expected 1 argument, got " << nrhs;
    mex_error(error_msg.str());
  }

  int device_id = static_cast<int>(mxGetScalar(prhs[0]));
  Caffe::SetDevice(device_id);
}

static void get_init_key(MEX_ARGS) {
  plhs[0] = mxCreateDoubleScalar(init_key);
}

static void init(MEX_ARGS) {
  if (nrhs != 3) {
    ostringstream error_msg;
    error_msg << "Expected 3 arguments, got " << nrhs;
    mex_error(error_msg.str());
  }

  char* param_file = mxArrayToString(prhs[0]);
  char* model_file = mxArrayToString(prhs[1]);
  char* phase_name = mxArrayToString(prhs[2]);

  Phase phase;
  if (strcmp(phase_name, "train") == 0) {
      phase = TRAIN;
  } else if (strcmp(phase_name, "test") == 0) {
      phase = TEST;
  } else {
    mex_error("Unknown phase.");
  }

  net_.reset(new Net<float>(string(param_file), phase));
  net_->CopyTrainedLayersFrom(string(model_file));

  mxFree(param_file);
  mxFree(model_file);
  mxFree(phase_name);

  init_key = random();  // NOLINT(caffe/random_fn)

  if (nlhs == 1) {
    plhs[0] = mxCreateDoubleScalar(init_key);
  }
}

static void reset(MEX_ARGS) {
  if (net_) {
    net_.reset();
    init_key = -2;
    LOG(INFO) << "Network reset, call init before use it again";
  }
}

static void forward(MEX_ARGS) {
  if (nrhs != 1) {
    ostringstream error_msg;
    error_msg << "Expected 1 argument, got " << nrhs;
    mex_error(error_msg.str());
  }

  plhs[0] = do_forward(prhs[0]);
}

static void backward(MEX_ARGS) {
  if (nrhs != 1) {
    ostringstream error_msg;
    error_msg << "Expected 1 argument, got " << nrhs;
    mex_error(error_msg.str());
  }

  plhs[0] = do_backward(prhs[0]);
}

static void is_initialized(MEX_ARGS) {
  if (!net_) {
    plhs[0] = mxCreateDoubleScalar(0);
  } else {
    plhs[0] = mxCreateDoubleScalar(1);
  }
}

static void read_mean(MEX_ARGS) {
    if (nrhs != 1) {
        mexErrMsgTxt("Usage: caffe('read_mean', 'path_to_binary_mean_file'");
        return;
    }
    const string& mean_file = mxArrayToString(prhs[0]);
    Blob<float> data_mean;
    LOG(INFO) << "Loading mean file from: " << mean_file;
    BlobProto blob_proto;
    bool result = ReadProtoFromBinaryFile(mean_file.c_str(), &blob_proto);
    if (!result) {
        mexErrMsgTxt("Couldn't read the file");
        return;
    }
    data_mean.FromProto(blob_proto);
    mwSize dims[4] = {data_mean.width(), data_mean.height(),
                      data_mean.channels(), data_mean.num() };
    mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    caffe_copy(data_mean.count(), data_mean.cpu_data(), data_ptr);
    mexWarnMsgTxt("Remember that Caffe saves in [width, height, channels]"
                  " format and channels are also BGR!");
    plhs[0] = mx_blob;
}

static int do_get_num_layers() {
  return net_->blobs().size();
}

static mxArray* do_get_layer(int L) {
  const shared_ptr< Blob<float> > layer_blob = net_->blobs()[L];
  mwSize dims[4] = {layer_blob->width(), layer_blob->height(), layer_blob->channels(), layer_blob->num()};
  mxArray* mx_layer = mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
  float* layer_ptr = reinterpret_cast<float*>(mxGetPr(mx_layer));
  
  switch (Caffe::mode()) {
    case Caffe::CPU:
      memcpy(layer_ptr, layer_blob->cpu_data(),
          sizeof(float) * layer_blob->count());
      break;
    // case Caffe::GPU:
    //  CUDA_CHECK(cudaMemcpy(layer_ptr, layer_blob->gpu_data(),
    //      sizeof(float) * layer_blob->count(), cudaMemcpyDeviceToHost));
   //  break;
    default:
      LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
  return mx_layer;
}

static mxArray* do_get_all_layers() {
  int n = do_get_num_layers();
  const mwSize dims[2] = {n, 1};
  mxArray* mx_layer_cells = mxCreateCellArray(2, dims);
  
  for(int i=0; i<n; ++i){
    mxSetCell(mx_layer_cells, i, do_get_layer(i));
  }
  
  return mx_layer_cells;
}

static void get_num_layers(MEX_ARGS) {
  plhs[0] = mxCreateDoubleScalar(do_get_num_layers());
}

static mxArray* do_get_layer_name(int L) {
  return mxCreateString(net_->blob_names()[L].c_str());
}

static mxArray* do_get_all_layer_names() {
  int n = do_get_num_layers();
  const mwSize dims[2] = {n, 1};
  mxArray* mx_layer_names = mxCreateCellArray(2, dims);
  
  for(int i=0; i<n; ++i){
    mxSetCell(mx_layer_names, i, do_get_layer_name(i));
  }
  
  return mx_layer_names;
}

static void get_layer(MEX_ARGS) {
  int L = static_cast<int>(mxGetScalar(prhs[0]));
  plhs[0] = do_get_layer(L);
  plhs[1] = do_get_layer_name(L);
}

static void get_all_layers(MEX_ARGS) {
  plhs[0] = do_get_all_layers();
  plhs[1] = do_get_all_layer_names();
}

static void get_all_layer_names(MEX_ARGS) {
  plhs[0] = do_get_all_layer_names();
}

static void get_layer_name(MEX_ARGS){
  int L = static_cast<int>(mxGetScalar(prhs[0]));
  plhs[0] = do_get_layer_name(L);
}

static void mxarray_to_blob_data(const mxArray* data, Blob<float>* blob) {
  mwSize dims[4] = {blob->width(), blob->height(),
                    blob->channels(), blob->num()};
  LOG(INFO) << dims[0] << " " << dims[1] << " " << dims[2] << " " << dims[3];
  CHECK_EQ_MEX(blob->count(), mxGetNumberOfElements(data),
    "blob->count() don't match with numel(data)");
  const float* const data_ptr =
      reinterpret_cast<const float* const>(mxGetPr(data));
  switch (Caffe::mode()) {
  case Caffe::CPU:
    caffe_copy(blob->count(), data_ptr, blob->mutable_cpu_data());
    break;
  case Caffe::GPU:
    caffe_copy(blob->count(), data_ptr, blob->mutable_gpu_data());
    break;
  default:
    LOG(FATAL) << "Unknown Caffe mode.";
    mexErrMsgTxt("Unknown caffe mode");
  }  // switch (Caffe::mode())
}

static void mxarray_to_blob_data(const mxArray* data,
    shared_ptr<Blob<float> > blob) {
  mxarray_to_blob_data(data, blob.get());
}

static void save_net(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
  if (net_) {
  char* model_file = mxArrayToString(prhs[0]);
  NetParameter net_param;
  net_->ToProto(&net_param, false);
  WriteProtoToBinaryFile(net_param, model_file);
  CheckFile(string(model_file));
  mxFree(model_file);
  } else {
    mexErrMsgTxt("Need to have a network to save");
  }
}

static void do_set_layer_weights(const mxArray* const layer_name,
    const mxArray* const mx_layer_weights) {
  const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
  const vector<string>& layer_names = net_->layer_names();

  char* c_layer_name = mxArrayToString(layer_name);
  LOG(INFO) << "Looking for: " << c_layer_name;

  for (unsigned int i = 0; i < layers.size(); ++i) {
    LOG(INFO) << layer_names[i];
    if (strcmp(layer_names[i].c_str(), c_layer_name) == 0) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }
      LOG(INFO) << "Found layer " << layer_names[i];
      CHECK_EQ_MEX(static_cast<unsigned int>(
        mxGetDimensions(mx_layer_weights)[0]),
        layer_blobs.size(), "Num of cells don't match layer_blobs.size");
      LOG(INFO) << "layer_blobs.size() = " << layer_blobs.size();
      for (unsigned int j = 0; j < layer_blobs.size(); ++j) {
        // internally data is stored as (width, height, channels, num)
        // where width is the fastest dimension
        const mxArray* const elem = mxGetCell(mx_layer_weights, j);
        mxarray_to_blob_data(elem, layer_blobs[j]);
      }
    }
  }
}

static void set_weights(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Given " << nrhs << " arguments expecting 1";
    mexErrMsgTxt("Wrong number of arguments");
  }
  const mxArray* const mx_weights = prhs[0];
  if (!mxIsStruct(mx_weights)) {
    mexErrMsgTxt("Input needs to be struct");
  }
  int num_layers = mxGetNumberOfElements(mx_weights);
  for (int i = 0; i < num_layers; ++i) {
    const mxArray* layer_name= mxGetField(mx_weights, i, "layer_names");
    const mxArray* weights= mxGetField(mx_weights, i, "weights");
    do_set_layer_weights(layer_name, weights);
  }
}


/** -----------------------------------------------------------------
 ** Available commands.
 **/
struct handler_registry {
  string cmd;
  void (*func)(MEX_ARGS);
};

static handler_registry handlers[] = {
  // Public API functions
  { "forward",            forward         },
  { "backward",           backward        },
  { "init",               init            },
  { "is_initialized",     is_initialized  },
  { "set_mode_cpu",       set_mode_cpu    },
  { "set_mode_gpu",       set_mode_gpu    },
  { "set_device",         set_device      },
  { "get_weights",        get_weights     },
  { "get_init_key",       get_init_key    },
  { "get_layer",          get_layer       },
  { "get_all_layers",     get_all_layers  },
  { "get_num_layers",     get_num_layers  },
  { "reset",              reset           },
  { "get_names",          get_all_layer_names},
  { "get_name",           get_layer_name  },
  { "read_mean",          read_mean       },
  { "save_net",           save_net        },
  { "set_weights",        set_weights     },
  // The end.
  { "END",                NULL            },
};


/** -----------------------------------------------------------------
 ** matlab entry point: caffe(api_command, arg1, arg2, ...)
 **/
void mexFunction(MEX_ARGS) {
  mexLock();  // Avoid clearing the mex file.
  if (nrhs == 0) {
    mex_error("No API command given");
    return;
  }

  { // Handle input command
    char *cmd = mxArrayToString(prhs[0]);
    bool dispatched = false;
    // Dispatch to cmd handler
    for (int i = 0; handlers[i].func != NULL; i++) {
      if (handlers[i].cmd.compare(cmd) == 0) {
        handlers[i].func(nlhs, plhs, nrhs-1, prhs+1);
        dispatched = true;
        break;
      }
    }
    if (!dispatched) {
      ostringstream error_msg;
      error_msg << "Unknown command '" << cmd << "'";
      mex_error(error_msg.str());
    }
    mxFree(cmd);
  }
}

