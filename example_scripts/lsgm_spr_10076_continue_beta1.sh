#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --account=def-janehowe
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=64000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=1-22:50     # DD-HH:MM:SS


module load StdEnv/2020 python/3.7 cuda cudnn  

SOURCEDIR=/home/shibin2/projects/def-janehowe/shibin2
DATADIR=/home/shibin2/projects/def-janehowe/shared_2022/dataset/EMPIAR-10076_downsampled/256
RESULT=/home/shibin2/projects/def-janehowe/shared_2022/output/10076_beta1

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


cp -rf /home/shibin2/projects/def-janehowe/shared_2022/scripts/DGP_SPR ./
cd DGP-SPR


python train_lsgm.py $DATADIR/particles.256.txt --poses $DATADIR/pose.pkl --ctf $DATADIR/ctf.pkl --zdim 10 --epochs 101 --root $RESULT --save '3/exp_lsgm_5e-4'  --vada_checkpoint $RESULT/3/exp_lsgm_5e-4/checkpoint.pt --cont_training --enc-dim 256 --enc-layers 3 --dec-dim 256 --dec-layers 3 --dropout 0.1 --batch_size 8 --num_scales_dae 2 --weight_decay_norm_vae 1e-2 --weight_decay_norm_dae 0. --num_channels_dae 8 --train_vae --num_cell_per_scale_dae 1 --learning_rate_dae 3e-4 --learning_rate_min_dae 3e-4 --train_ode_solver_tol 1e-5  --sde_type vpsde --iw_sample_p ll_iw --num_process_per_node 1 --dae_arch ncsnpp --embedding_scale 50 --mixing_logit_init -3 --warmup_epochs 0 --lazy --disjoint_training --iw_sample_q ll_iw --iw_sample_p drop_sigma2t_iw --embedding_dim 64 --beta 1 --learning_rate_vae 5e-4 --weight_decay 0 --uninvert-data





