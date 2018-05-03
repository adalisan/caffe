#ifndef CAFFE_MIL_DATA_LAYER_HPP_
#define CAFFE_MIL_DATA_LAYER_HPP_

#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/db.hpp"


namespace caffe {



/**
* @brief Uses a text file which specifies the image names, and a hdf5 file for the labels.
*        Note that each image can have multiple positive labels.
*
* TODO(dox): thorough documentation for Forward and proto params.
*/
 template <typename Dtype>
     class MILDataLayer : public BasePrefetchingDataLayer<Dtype> {
      public:
        explicit MILDataLayer(const LayerParameter& param)
                : BasePrefetchingDataLayer<Dtype>(param) {}
        virtual ~MILDataLayer();
        virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
                    
        virtual inline const char* type() const { return "MILData"; }
        virtual inline int ExactNumBottomBlobs() const { return 0; }
        virtual inline int ExactNumTopBlobs() const { return 2; }
                            
    protected:
        virtual unsigned int PrefetchRand();
        virtual void InternalThreadEntry();
        virtual void load_batch(Batch<Dtype>*batch);
        
        int num_images_;
        unsigned int counter_;
        shared_ptr<Caffe::RNG> prefetch_rng_;
        vector< std::pair<std::string, std::string > > image_database_;
        hid_t label_file_id_;
        
            vector<float> mean_value_;
            Blob<Dtype> label_blob_;
        };
}  // namespace caffe

#endif  // CAFFE_MIL_DATA_LAYER_HPP_
