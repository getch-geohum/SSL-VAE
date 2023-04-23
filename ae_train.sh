#!/bin/bash
#for fold in carpet grid leather tile wood all
#do
python ae_train.py\
	--model=dis_ssvae\
	--exp=dis_ssvae\
	--num_epochs=51\
	--lr=1e-4\
	--img_size=256\
	--batch_size=16\
	--batch_size_test=8\
	--latent_img_size=32\
    --z_dim=6\
    --lamda=0.9\
    --nb_channels=4\
    --fake_dataset_size=500\
    --params_id=100\
    --dst_dir=./outputs/\
    --data_dir=../data/\
    --data=Minawao_june_2016,Minawao_feb_2017,Tza_oct_2016\
    --max_beta=1\
    --dataset=camp\
	--force_train\
	--with_mask\
	--with_prob\
    #--with_condition\
	#--anneal_beta\
	#--anneal_cyclic\
	#--min_beta=0\
    #--cycle=6\
    #--ratio=0.5\

#done
# dataset could be either "camp" or "mvtec"
# --ndvi_treshold=0.1\
# --intensity_treshold=145\
# --brightness_treshold=0\
# --contrast_treshold=0\
# --func=NDVI\
# --equalize\
