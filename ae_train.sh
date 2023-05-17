#!/bin/bash
#for fold in carpet grid leather tile wood all
#do
python ae_train.py\
	--model=dis_ssvae\
	--exp=dis_ssvae\
	--num_epochs=100\
	--lr=1e-4\
	--img_size=256\
	--batch_size=16\
	--batch_size_test=8\
	--latent_img_size=32\
	--z_dim=54\
	--lamda=0.9\
	--nb_channels=4\
	--fake_dataset_size=100\
	--params_id=100\
	--dst_dir=D:/DATA/MVCamp_dis/outputs_34full_m6\
	--data_dir=D:/DATA/rawdata/camp\
	--data=Deghale_Apr_2017,Kule_tirkidi_jun_2018,Kule_tirkidi_marc_2017,kutupalong_sept_2017,Minawao_feb_2017,Minawao_june_2016,Nguyen_march_2017,Tza_oct_2016,Zamzam_april_2022\
	--max_beta=1\
	--dataset=camp\
	--z_dim_constrained=6\
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
