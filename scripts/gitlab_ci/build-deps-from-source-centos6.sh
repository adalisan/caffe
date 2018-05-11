#!/bin/bash
BLD_DIR=$1
function download {
    if [[ ! -e $2 ]]; then
        echo "Downloading $1"
        curl -L $1 -o $2
    fi
}

PLATFORM=linux-centos6



if [[ $PLATFORM == windows* ]]; then
    echo TODO
else
    GLOG=0.3.4
	GFLAGS=2.1.2
    PROTO=2.6.1
    LEVELDB=1.18
    SNAPPY=1.1.3
    LMDB=0.9.15
    BOOST=1_61_0 # also change the download link, 1.58 incompatible with OSX
    HDF5=1.8.20
    BLAS=0.2.14
    MAKEJ="${MAKEJ:-4}"
    if [[ ! -e v$GLOG.tar.gz ]]; then 
    download https://github.com/google/glog/archive/v$GLOG.tar.gz v$GLOG.tar.gz
    fi
   if [[ ! -e v$GFLAGS.tar.gz ]]; then 
    download https://github.com/gflags/gflags/archive/v$GFLAGS.tar.gz v$GFLAGS.tar.gz
    fi
   if [[ ! -e protobuf-$PROTO.tar.gz ]]; then 
    download https://github.com/google/protobuf/releases/download/v$PROTO/protobuf-$PROTO.tar.gz protobuf-$PROTO.tar.gz
    fi
   if [[ ! -e  v$LEVELDB.tar.gz ]]; then 
    download https://github.com/google/leveldb/archive/v$LEVELDB.tar.gz v$LEVELDB.tar.gz
    fi
   if [[ ! -e snappy-$SNAPPY.tar.gz ]]; then 
    download https://github.com/google/snappy/releases/download/$SNAPPY/snappy-$SNAPPY.tar.gz snappy-$SNAPPY.tar.gz
    fi
   if [[ ! -e LMDB_$LMDB.tar.gz ]]; then 
    download https://github.com/LMDB/lmdb/archive/LMDB_$LMDB.tar.gz LMDB_$LMDB.tar.gz
    fi
   if [[ ! -e boost_$BOOST.tar.gz ]]; then 
    download http://iweb.dl.sourceforge.net/project/boost/boost/1.60.0/boost_$BOOST.tar.gz boost_$BOOST.tar.gz
    fi
   if [[ ! -e hdf5-$HDF5.tar.gz ]]; then 
    download https://www.hdfgroup.org/ftp/HDF5/current18/src/hdf5-$HDF5.tar.gz hdf5-$HDF5.tar.gz
    fi
   if [[ ! -e v$BLAS.tar.gz ]]; then 
    download https://github.com/xianyi/OpenBLAS/archive/v$BLAS.tar.gz v$BLAS.tar.gz
    fi
    if [[ ! -e $PLATFORM ]]; then
        mkdir -p $PLATFORM
    fi
    cd $PLATFORM
    INSTALL_PATH=`pwd`
    mkdir -p include
    mkdir -p lib

    echo "Decompressing archives"
    tar --totals -xzf ../v$GLOG.tar.gz
    tar --totals -xzf ../v$GFLAGS.tar.gz
    tar --totals -xzf ../protobuf-$PROTO.tar.gz
    tar --totals -xzf ../v$LEVELDB.tar.gz
    tar --totals -xzf ../snappy-$SNAPPY.tar.gz
    tar --totals -xzf ../LMDB_$LMDB.tar.gz
    tar --totals -xzf ../boost_$BOOST.tar.gz
    tar --totals -xzf ../hdf5-$HDF5.tar.gz
    tar --totals -xzf ../v$BLAS.tar.gz
fi

cd glog-$GLOG
./configure --prefix=$INSTALL_PATH
make -j $MAKEJ >> /dev/null
make install  >> /dev/null
cd ..

cd gflags-$GFLAGS
mkdir -p build
cd build
export CXXFLAGS="-fPIC" && cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH ..
make -j $MAKEJ >> /dev/null
make install  >> /dev/null
cd ../..

cd protobuf-$PROTO
./configure --prefix=$INSTALL_PATH
make -j $MAKEJ >> /dev/null
make install  >> /dev/null
cd ..

cd leveldb-$LEVELDB
make -j $MAKEJ
cp -a libleveldb.* $INSTALL_PATH/lib
cp -a include/leveldb $INSTALL_PATH/include/
cd ..

cd snappy-$SNAPPY
./configure --prefix=$INSTALL_PATH
make -j $MAKEJ >> /dev/null
make install  >> /dev/null
cd ..

cd lmdb-LMDB_$LMDB/libraries/liblmdb
make -j $MAKEJ  >> /dev/null
cp -a lmdb.h ../../../include/
cp -a liblmdb.so ../../../lib/
cd ../../..

cd boost_$BOOST
./bootstrap.sh --with-libraries=system,thread,python  
./b2 install --prefix=$INSTALL_PATH  >> /dev/null
cd ..

cd hdf5-$HDF5
./configure --prefix=$INSTALL_PATH
make -j $MAKEJ  >> /dev/null
make install  >> /dev/null
cd ..

# OSX has Accelerate
if [[ $PLATFORM != macosx-* ]]; then
    # blas (requires fortran, e.g. sudo yum install gcc-gfortran)
	cd OpenBLAS-$BLAS
    # CentOS compiler version can't compile AVX2 instructions, TODO update compiler
	make -j $MAKEJ NO_AVX2=1
	make install PREFIX=$INSTALL_PATH
	cd ..
fi
