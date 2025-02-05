#!/bin/bash
#for fold in carpet grid leather tile wood all
#do
python ae_train.py\
	--model=dis_ssvae\
	--exp=dis_ssvae\
	--num_epochs=100\
	--lr=0.0001\
	--img_size=256\
	--batch_size=16\
	--batch_size_test=16\
	--latent_img_size=32\
	--z_dim=18\
	--lamda=0.9\
	--nb_channels=4\
	--fake_dataset_size=400\
	--params_id=100\
	--dst_dir=/home/getch/ssl/SSL_VAE/dis_ssvae_l0_2r\
	--data_dir=/home/getch/DATA/VAE/data\
	--data=Deghale_Apr_2017,Kule_tirkidi_marc_2017,Minawao_feb_2017,Nguyen_march_2017,Zamzam_april_2022,Kule_tirkidi_jun_2018,kutupalong_sept_2017,Minawao_june_2016,Tza_oct_2016\
	--max_beta=1\
	--dataset=camp\
	--z_dim_constrained=2\
	--force_train\
	--with_mask\
	--with_prob\

    #--with_condition\
	#--anneal_beta\
	#--anneal_cyclic\
	#--min_beta=0\
    #--cycle=6\
    #--ratio=0.5\

    # 
# outputs_34full_m4mm_mean_mask_m+

#done
# dataset could be either "camp" or "mvtec"
# --ndvi_treshold=0.1\
# --intensity_treshold=145\
# --brightness_treshold=0\
# --contrast_treshold=0\
# --func=NDVI\
# --equalize\
