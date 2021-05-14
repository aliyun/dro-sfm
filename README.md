## DRO: Deep Recurrent Optimizer for Structure-from-Motion

This is the official PyTorch implementation code for DRO-sfm. For technical details, please refer to:

**DRO: Deep Recurrent Optimizer for Structure-from-Motion** <br />
Xiaodong Gu\*, Weihao Yuan\*, Zuozhuo Dai, Chengzhou Tang, Siyu Zhu, Ping Tan <br />
**[[Paper](https://arxiv.org/abs/2103.13201)]** <br />

<p float="left">
  <img src="/media/figs/demo_kitti.gif" width="400" />
  <img src="/media/figs/demo_scannet.gif" width="400" /> 
</p>

## Bibtex
If you find this code useful in your research, please cite:

```
@article{gu2021dro,
  title={DRO: Deep Recurrent Optimizer for Structure-from-Motion},
  author={Gu, Xiaodong and Yuan, Weihao and Dai, Zuozhuo and Tang, Chengzhou and Zhu, Siyu and Tan, Ping},
  journal={arXiv preprint arXiv:2103.13201},
  year={2021}
}
```

## Install
+ We recommend using [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker) to have a reproducible environment. 

```bash
git clone https://github.com/aliyun/dro-sfm.git
cd dro-sfm
sudo make docker-build
sudo make docker-start-interactive
```
You can also download the built docker directly from [dro-sfm-image.tar](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/dro-sfm/dro-sfm-image.tar)

```bash
docker load < dro-sfm-image.tar
```

+ If you would rather not use docker, you could create an environment via following the steps in the Dockerfile.

```bash
# Environment variables
export PYTORCH_VERSION=1.4.0
export TORCHVISION_VERSION=0.5.0
export NCCL_VERSION=2.4.8-1+cuda10.1
export HOROVOD_VERSION=65de4c961d1e5ad2828f2f6c4329072834f27661
# Install NCCL
sudo apt-get install libnccl2=${NCCL_VERSION} libnccl-dev=${NCCL_VERSION}

# Install Open MPI
mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
    tar zxf openmpi-4.0.0.tar.gz && \
    cd openmpi-4.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Install PyTorch
pip install torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} && ldconfig

# Install horovod (for distributed training)
sudo ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir git+https://github.com/horovod/horovod.git@${HOROVOD_VERSION} && sudo ldconfig
```

To verify that the environment is setup correctly, you can run a simple overfitting test:

```bash
# download a tiny subset of KITTI
cd dro-sfm
curl -s https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/dro-sfm/datasets/KITTI_tiny.tar | tar xv -C /data/datasets/kitti/
# in docker
./run.sh "python scripts/train.py configs/overfit_kitti_mf_gt.yaml" log.txt
```

## Datasets
Datasets are assumed to be downloaded in `/data/datasets/<dataset-name>` (can be a symbolic link).

### KITTI
The KITTI (raw) dataset used in our experiments can be downloaded from the [KITTI website](http://www.cvlibs.net/datasets/kitti/raw_data.php).
For convenience, you can download data from [packnet](https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/datasets/KITTI_raw.tar.gz) or [here](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/dro-sfm/datasets/KITTI_raw.tar)

### Tiny KITTI
For simple tests, you can download a "tiny" version of [KITTI](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/dro-sfm/datasets/KITTI_tiny.tar):


### Scannet
The Scannet (raw) dataset used in our experiments can be downloaded from the [Scannet website](http://www.scan-net.org). 
For convenience, you can download data from [here](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/dro-sfm/datasets/scannet.tar)


### DeMoN
Download [DeMoN](https://github.com/lmb-freiburg/demon/tree/master/datasets).

```bash
bash download_traindata.sh
python ./dataset/preparation/preparedata_train.py
bash download_testdata.sh
python ./dataset/preparation/preparedata_test.py
```

## Training
Any training, including fine-tuning, can be done by passing either a `.yaml` config file or a `.ckpt` model checkpoint to [scripts/train.py](./scripts/train.py):

```bash
# kitti, checkpoints will saved in ./results/mdoel/
./run.sh 'python scripts/train.py  configs/train_kitti_mf_gt.yaml' logs/kitti_sup.txt
./run.sh 'python scripts/train.py  configs/train_kitti_mf_selfsup.yaml' logs/kitti_selfsup.txt 

# scannet
./run.sh 'python scripts/train.py  configs/train_scannet_mf_gt_view3.yaml' logs/scannet_sup.txt
./run.sh 'python scripts/train.py  configs/train_scannet_mf_selfsup_view3.yaml' logs/scannet_selfsup.txt
./run.sh 'python scripts/train.py  configs/train_scannet_mf_gt_view5.yaml' logs/scannet_sup_view5.txt

# demon
./run.sh 'python scripts/train.py  configs/train_demon_mf_gt.yaml' logs/demon_sup.txt
```


## Evaluation

```bash
python scripts/eval.py --checkpoint <checkpoint.ckpt> [--config <config.yaml>]
# example:kitti, results will be saved in results/depth/
python scripts/eval.py --checkpoint ckpt/outdoor_kitti.ckpt --config configs/train_kitti_mf_gt.yaml

```

You can also directly run inference on a single image or video:

```bash
# video or folder
# indoor-scannet 
 python scripts/infer_video.py --checkpoint ckpt/indoor_sacnnet.ckpt --input /path/to/video or folder --output /path/to/save_folder --sample_rate 1 --data_type scannet --ply_mode 
 # indoor-general
python scripts/infer_video.py --checkpoint ckpt/indoor_sacnnet.ckpt --input /path/to/video or folder --output /path/to/save_folder --sample_rate 1 --data_type general --ply_mode

# outdoor
 python scripts/infer_video.py --checkpoint ckpt/outdoor_kitti.ckpt --input /path/to/video or folder --output /path/to/save_folder --sample_rate 1 --data_type kitti --ply_mode 

# image
python scripts/infer.py --checkpoint <checkpoint.ckpt> --input <image or folder> --output <image or folder>
```


## Models

| Model | Abs.Rel. | Sqr.Rel | RMSE | RMSElog | a1 | a2 | a3| SILog| L1_inv| rot_ang| t_ang| t_cm| 
| :--- | :---: | :---: | :---: |  :---: |  :---: |  :---: |  :---: |  :---: |  :---: |  :---: |  :---: |  :---: |
|[Kitti_sup](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/dro-sfm/models/outdoor_kitti.ckpt) | 0.045 | 0.193 | 2.570 | 0.080 | 0.971 | 0.994 | 0.998| 0.079 | 0.003 | -| -| -|
|[Kitti_selfsup](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/dro-sfm/models/outdoor_kitti_selfsup.ckpt) | 0.053 |0.346 | 3.037 | 0.102 | 0.962 | 0.990| 0.996|0.101 | 0.004 | -| -| -|
|[scannet_sup](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/dro-sfm/models/indoor_scannet.ckpt) | 0.053 | 0.017 | 0.165 | 0.080 | 0.967 | 0.994 | 0.998| 0.078 | 0.033| 0.472| 9.297| 1.160|
|[scannet_sup(view5)](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/dro-sfm/models/indoor_scannet_view5.ckpt) |0.047 |0.014 | 0.151 | 0.072 | 0.976 | 0.996 | 0.999| 0.071 | 0.030 | 0.456| 8.502| 1.163|
|[scannet_selfsup](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/dro-sfm/models/indoor_scannet_selfsup.ckpt) | 0.143 | 0.345 | 0.656 | 0.274 | 0.896 | 0.954 | 0.969|0.272 | 0.106 | 0.609| 10.779| 1.393 |



## Acknowledgements
Thanks to Toyota Research Institute for opening source of excellent work [packnet-sfm](https://github.com/TRI-ML/packnet-sfm). Thanks to Zachary Teed for opening source of his excellent work [RAFT](https://github.com/princeton-vl/RAFT).
