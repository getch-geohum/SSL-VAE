#!/bin/bash

python ae_train.py\
	--model=mv_ae\
        --exp=mv_ae\
        --num_epochs=100\
        --lr=1e-4\
        --img_size=512\
        --batch_size=8\
        --batch_size_test=4\
        --latent_img_size=32\
        --z_dim=256\
        --lamda=0.9\
        --nb_channels=3\
        --params_id=100\
        --ndvi_treshold=0.1\
        --intensity_treshold=145\
        --brightness_treshold=0\
        --contrast_treshold=0\
        --func=NDVI\
        --dst_dir=D:/DATA/MVTec/carpet\
        --data_dir=D:/DATA/rawdata/mvtec\
        --max_beta=0.1\
	--min_beta=0.00001\
	--dataset=mvtec\
	--texture=carpet\
        --with_mask\
        --force_train\
        --equalize\
	--anneal_beta\
	--anneal2descend\
	--with_prob\

# dataset could be either "camp" or "mvtec"
