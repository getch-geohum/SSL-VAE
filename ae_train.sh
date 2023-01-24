#!/bin/bash

python3 ae_train.py\
	--exp=vae_vae\
	--num_epochs=200\
	--lr=1e-4\
	--img_size=256\
	--batch_size=16\
	--batch_size_test=8\
	--latent_img_size=8\
	--z_dim=256\
	--lamda=0.9\
	--nb_channels=4\
	--params_id=100\
	--ndvi_treshold=0.1\
	--intensity_treshold=120\
	--func=NDVI\
	--dst_dir=/home/getch/ssl/DATA/UAE/OUTS/Deghale_Apr_2017\
        --data_dir=/home/getch/DATA/VAE/data/Deghale_Apr_2017\
	--force_train\
	#--equalize\

# equalize should be active if func=intensity and if we want to run histogra, equaliation
