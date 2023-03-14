#!/bin/bash
#for fold in carpet grid leather tile wood all
#do
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
        --nb_channels=5\
        --fake_dataset_size=1040\
        --params_id=100\
        --dst_dir=D:/DATA/MVCamp/FEATURSPACE__\
        --data_dir=D:/DATA/rawdata/campp\
        --data=all\
        --max_beta=1\
	--min_beta=0\
        --cycle=6\
        --ratio=0.5\
        --dataset=camp\
        --texture=carpet\
	--with_mask\
	--force_train\
	--with_prob\
	--anneal_beta\
	--anneal_cyclic\

#done
# dataset could be either "camp" or "mvtec"
# --ndvi_treshold=0.1\
# --intensity_treshold=145\
# --brightness_treshold=0\
# --contrast_treshold=0\
# --func=NDVI\
# --equalize\
