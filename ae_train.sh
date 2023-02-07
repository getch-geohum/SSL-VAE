#!/bin/bash

python3 ae_train.py\
	--model=ssvae\
	--exp=vae_vae\
	--num_epochs=200\
	--lr=1e-4\
	--img_size=256\
	--batch_size=16\
	--batch_size_test=8\
	--latent_img_size=32\
	--z_dim=256\
	--lamda=0.9\
	--nb_channels=4\
	--params_id=100\
	--ndvi_treshold=0.10\
	--intensity_treshold=120\
	--brightness_treshold=7\
	--contrast_treshold=0\
	--func=grayIntensity\
	--dst_dir=/home/getch/ssl/DATA/SSL_VAE_aaaa+/OUTS/Tza_oct_2016\
        --data_dir=/home/getch/DATA/VAE/data/Tza_oct_2016\
	--force_train\
	#--equalize\

# equalize should be active if func=intensity and if we want to run histogra, equaliation
# if contrast_treshold is greater than zero, brightness_treshold should be zero and vice versa
# if brightness treshold is zer0, contrast_treshold should be 11      
# these tresholds were only tested in Nguyen_march_2017
