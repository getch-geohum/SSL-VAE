#!/bin/bash

python3 ae_test.py\
    --exp=vae_vae\
    --model=ssvae\
    --num_epochs=100\
    --lr=1e-4\
    --img_size=256\
    --batch_size=16\
    --batch_size_test=8\
    --latent_img_size=32\
    --z_dim=256\
    --lamda=0.9\
    --nb_channels=4\
    --params_id=100\
    --dst_dir=/home/getch/ssl/DATA/SSL_VAE_a/OUTS/Minawao_feb_2017\
    --data_dir=/home/getch/DATA/VAE/data/Minawao_feb_2017\

