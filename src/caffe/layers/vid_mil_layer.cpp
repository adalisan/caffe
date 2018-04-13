// Copyright 2014 BVLC and contributors.

#include <vector>
#include <cmath>
#include <cfloat>
#include <iostream>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
void VidMILLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 1) << "MIL Layer takes a single blob as input.";
  CHECK_EQ(top.size(), 1) << "MIL Layer takes a single blob as output.";
}
template <typename Dtype>
void VidMILLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, 
  const vector<Blob<Dtype>*>& top){
  top[0]->Reshape(1, bottom[0]->channels(), 1, 1);
}

template <typename Dtype>
void VidMILLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int offset;
  channels_   = bottom[0]->channels();
  num_images_ = bottom[0]->num();
  height_     = bottom[0]->height();
  width_      = bottom[0]->width();
  float pool_size = num_images_ * height_ * width_;


  for(int j = 0; j < channels_; j++){
    Dtype prob, max_prob;
    switch (this->layer_param_.mil_param().type()) {
      case VidMILParameter_MILType_MAX:
	prob = -FLT_MAX; 
	for(int i = 0; i < num_images_; i++){
	  for(int k = 0; k < height_; k++){
	    for(int l = 0; l < width_; l++){
	      offset = bottom[0]->offset(i, j, k, l);
	      prob = max(prob, bottom_data[offset]);
	    }
	  }
	}
	top_data[j] = prob;
	break;

      case VidMILParameter_MILType_AVE:
	prob = 0.; 
	for(int i = 0; i < num_images_; i++){
	  for(int k = 0; k < height_; k++){
	    for(int l = 0; l < width_; l++){
	      offset = bottom[0]->offset(i, j, k, l);
	      prob += bottom_data[offset];
	    }
	  }
	}
	top_data[j] = prob/pool_size;
	break;

      case VidMILParameter_MILType_NOR:
	prob = 1.; max_prob = -FLT_MAX; 
	for(int i = 0; i < num_images_; i++){
	  for(int k = 0; k < height_; k++){
	    for(int l = 0; l < width_; l++){
	      offset = bottom[0]->offset(i, j, k, l);
	      prob = prob*(1. - bottom_data[offset]);
	      max_prob = max(max_prob, bottom_data[offset]);
	      CHECK_LE(bottom_data[offset], 1.) << "input mil_prob not <= 1";
	      CHECK_GE(bottom_data[offset], 0.) << "input mil_prob not >= 0";
	    }
	  }
	}
	top_data[j] = max(Dtype(1.) - prob, max_prob);
	CHECK_LE(top_data[j], 1.) << "mil_prob not <= 1";
	CHECK_GE(top_data[j], 0.) << "mil_prob not >= 0";
	break;
    }
  }
}

template <typename Dtype>
void VidMILLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  int offset;
  float pool_size = num_images_ * height_ * width_;

  if(propagate_down[0]){
    for(int j = 0; j < channels_; j++){
      for(int i = 0; i < num_images_; i++){
        for(int k = 0; k < height_; k++){
          for(int l = 0; l < width_; l++){
	    offset = bottom[0]->offset(i, j, k, l);
            switch (this->layer_param_.mil_param().type()) {
              case VidMILParameter_MILType_MAX:
                bottom_diff[offset] =
                  top_diff[j] * (top_data[j] == bottom_data[offset]);
                break;
	      case VidMILParameter_MILType_AVE:
                bottom_diff[offset] =
                  top_diff[j] / pool_size;
                break;
              case VidMILParameter_MILType_NOR:
                bottom_diff[offset] = top_diff[j] * 
                  min(Dtype(1.),((1-top_data[j])/(1-bottom_data[offset]))); 
                break;
            }
	  }
        } 
      }
    }
  }
}

INSTANTIATE_CLASS(VidMILLayer);
REGISTER_LAYER_CLASS(VidMIL);
}  // namespace caffe
