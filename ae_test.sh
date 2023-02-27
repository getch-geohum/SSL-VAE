#!/bin/bash

python ae_test.py\
	--exp=mv_ae\
    	--model=mv_ae\
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
    	--dst_dir=D:/DATA/mvtec/carpet\
    	--data_dir=D:/DATA/rawdata/mvtec\
    	--dataset=mvtec\
	--texture=carpet\
	--with_mask\
    	--with_prob\
    	--save_preds\

