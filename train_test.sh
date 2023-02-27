#!/bin/bash

for i in 0.001 0.01 0.1
do
	python ae_train.py\
		--model=ss_cvae\
        	--exp=ss_cvae\
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
        	--ndvi_treshold=0.3\
        	--intensity_treshold=145\
        	--brightness_treshold=0\
        	--contrast_treshold=0\
        	--func=NDVI\
        	--dst_dir=D:/DATA/Optim_inc/outs_optim_$i/Minawao_feb_2017\
        	--data_dir=D:/DATA/rawdata/Minawao_feb_2017\
        	--max_beta=$i\
		--min_beta=0.0001\
		--dataset=camp\
		--texture=carpet\
        	--with_mask\
        	--with_prob\
        	--force_train\
        	--equalize\
	        --anneal_beta\
		--anneal2descend\

        python ae_test.py\
                --exp=ss_cvae\
                --model=ss_cvae\
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
                --dst_dir=D:/DATA/Optim_inc/outs_optim_$i/Minawao_feb_2017\
                --data_dir=D:/DATA/rawdata/Minawao_feb_2017\
                --with_mask\
                --with_prob\
                --save_preds\

	done
