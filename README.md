# Install TVM
```
git clone --recursive https://github.com/kumazzu/tensor-program-predictor.git
cd tensor-program-predictor
```

## Install pre-requisites
I compiled TVM using clang+llvm-12.0.0-x86_64-linux-gnu-ubuntu-16.04 and cmake-3.18.0-rc1-Linux-x86_64. Download llvm and cmake of appropriate versions and configure the environment variables.

Edit build/config.cmake
```
cd /path/to/tensor-program-predictor
mkdir build
cp cmake/config.cmake build
```
Change set(USE_CUDA OFF) to set(USE_CUDA ON) and set(USE_LLVM OFF) to set(USE_LLVM ON)

## build tvm
```
cd build
cmake ..
make -j $(($(nproc) + 1))
```

## set the environment variable
Append environment variable to `~/.bashrc.`
```
export TVM_HOME=/path/to/tensor-program-predictor
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
```

## Python dependencies
```
pip install tornado psutil xgboost cloudpickle decorator pytest
pip install -r requirements.txt
```

# Sampling train tasks
Run samply.py to sample 20 tuning tasks by default. If you want to change the number of training tasks, adjust the parameter on line 116 of samply.py.
```
cd /path/to/tensor-program-predictor/dataset_generate
python sampling.py
```

# Training TPP cost model 
```
cd /path/to/tensor-program-predictor/train_model
python train.py --model tpp/transformer --dataset_dir <dataset path (e.g. /root/ost/dataset_generate)> --save_dir TPP/OST
```

# Evaluating
```
cd /path/to/tensor-program-predictor/eval
python tune.py --model_type tpp/transformer --model_path ../model_best.pth (or your own model) --network resnet-18 --n_trial 96 --save TPP/OST
python tune.py --model_type tpp/transformer --model_path ../model_best.pth (or your own model) --network mobilenetv2 --n_trial 160 --save TPP/OST
python tune.py --model_type tpp/transformer --model_path ../model_best.pth (or your own model) --network vgg-16 --n_trial 96 --save TPP/OST
python tune.py --model_type tpp/transformer --model_path ../model_best.pth (or your own model) --network densenet-121 --n_trial 160 --save TPP/OST
python tune.py --model_type tpp/transformer --model_path ../model_best.pth (or your own model) --network efficientnet --n_trial 160 --save TPP/OST
python tune.py --model_type tpp/transformer --model_path ../model_best.pth (or your own model) --network squeezenet_v1.0 --n_trial 96 --save TPP/OST
```

## Hardware dependencies
NVIDIA GeForce RTX 2080Ti and Intel(R) Xeon(R) Platinum 8255C CPU are recommended.

## Software dependencies
TPP is tested on Ubuntu 18.04 x86-64 system with CUDA 10.2.