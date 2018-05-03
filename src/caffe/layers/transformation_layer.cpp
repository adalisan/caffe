#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/transformation_layer.hpp"


namespace caffe {

template <typename Dtype>
void TransformationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 2) << "the number of bottoms should be 2";
  CHECK_EQ(top.size(), 1) << "the number of tops should be 1";
  CHECK_EQ(bottom[0]->num(), 1) << "batch size should be 1";

  CHECK_EQ(bottom[0]->num(),bottom[1]->num());
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
}

template <typename Dtype>
void TransformationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->shape());
}

template <typename Dtype>
void TransformationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype *bottom_data = bottom[0]->cpu_data();
  const Dtype *T_data = bottom[1]->cpu_data();
  num_classes_ = bottom[0]->shape(1);  
  Dtype *top_data = top[0]->mutable_cpu_data();  
  for(int c=0; c<num_classes_; c++) {
    const int offset = c*4;
    top_data[c] = T_data[offset+3]*bottom_data[c] + T_data[offset+2]*(1.0-bottom_data[c]);
  }
}

template <typename Dtype>
void TransformationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  num_classes_ = bottom[0]->shape(1);
  const Dtype *top_diff = top[0]->cpu_diff();
  const Dtype *bottom_data = bottom[0]->cpu_data();
  const Dtype *T_data = bottom[1]->cpu_data();

  if (propagate_down[0]) {
    Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
    for(int c=0; c<num_classes_; c++) {
      const int offset = c*4;
      bottom_diff[c] = top_diff[c]*(T_data[offset+3]-T_data[offset+2]);
    }
  }

  if (propagate_down[1]) {
    Dtype *Q_diff = bottom[1]->mutable_cpu_diff();                                                         
    caffe_set(bottom[1]->count(), Dtype(0), Q_diff);      
    for(int c=0; c<num_classes_; c++) {
      const int offset = c*4;
      Q_diff[offset+2] = top_diff[c]*(1.0-bottom_data[c]); 
      Q_diff[offset+3] = top_diff[c]*bottom_data[c]; 
    }
  } 
}

INSTANTIATE_CLASS(TransformationLayer);
REGISTER_LAYER_CLASS(Transformation);

}  // namespace caffe
