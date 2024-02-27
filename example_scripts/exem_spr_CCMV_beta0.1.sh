#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --account=def-janehowe
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=64000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=1-22:50     # DD-HH:MM:SS


module load StdEnv/2020 python/3.7 cuda cudnn  

SOURCEDIR=/home/shibin2/projects/def-janehowe/shibin2
DATADIR=/home/shibin2/projects/def-janehowe/shared_2022/dataset/CCMV
RESULT=/home/shibin2/projects/def-janehowe/shared_2022/output/CCMV2

# Prepare virtualenv
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip3 install --no-index --upgrade pip
pip3 install --no-index torch==1.8.0
pip3 install --no-index pillow
pip3 install --no-index matplotlib
pip3 install --no-index tensorboard
pip3 install --no-index tensorboardX
pip3 install --no-index scipy
pip3 install --no-index ninja
pip3 install --no-index pandas
pip install --no-index $SOURCEDIR/wheels/torchdiffeq-0.2.2-py3-none-any.whl
# pip install --no-index $SOURCEDIR/wheels/torchvision-0.9.0-py3-none-any.whl
pip install --no-index torchvision

cp -rf /home/shibin2/projects/def-janehowe/shared_2022/scripts/DGP-SPR ./
cd DGP-SPR

python train_exemplar.py $DATADIR/particles.256.mrcs  --poses $DATADIR/pose.pkl --ctf $DATADIR/ctf.pkl --zdim 10 -n 101 --root $RESULT --save 'exp_exemplar'  --enc-dim 256 --enc-layers 3 --dec-dim 256 --dec-layers 3  --amp --lazy --lr 0.00005 --beta 0.1 --checkpoint 5 --batch-size 8 --prior 'exemplar' --number-cachecomponents 5000 --approximate-prior --log-interval 10000




