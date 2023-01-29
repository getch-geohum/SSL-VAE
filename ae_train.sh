#!/bin/bash

python3 ae_train.py\
	--model=ssvae\
	--exp=vae_vae\
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
	--ndvi_treshold=0.1\
	--intensity_treshold=120\
	--func=NDVI\
	--dst_dir=/home/getch/ssl/DATA/SSL_VAE/OUTS/Minawao_june_2016\
        --data_dir=/home/getch/DATA/VAE/data/Minawao_june_2016\
	--force_train\
	#--equalize\

# equalize should be active if func=intensity and if we want to run histogra, equaliation
