#!/bin/bash


python cb_ae_train.py\
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
	--func=grayIntensity\
	--dst_dir=D:/DATA/outsdata_anneal/Kule_tirkidi_marc_2017\
        --data_dir=D:/DATA/rawdata/Kule_tirkidi_marc_2017\
        --max_beta=0.01\
	--with_mask\
        --with_prob\
	--force_train\
	--equalize\
	--anneal_beta\

python cb_ae_train.py\
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
	--intensity_treshold=120\
	--brightness_treshold=0\
	--contrast_treshold=0\
	--func=NDVI\
	--dst_dir=D:/DATA/outsdata_anneal/Minawao_feb_2017\
        --data_dir=D:/DATA/rawdata/Minawao_feb_2017\
        --max_beta=0.01\
	--with_mask\
        --with_prob\
	--force_train\
	--anneal_beta\
	#--equalize\

python cb_ae_train.py\
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
	--intensity_treshold=120\
	--brightness_treshold=6.5\
	--contrast_treshold=0\
	--func=grayIntensity\
	--dst_dir=D:/DATA/outsdata_anneal/Tza_oct_2016\
        --data_dir=D:/DATA/rawdata/Tza_oct_2016\
        --max_beta=0.01\
	--with_mask\
        --with_prob\
	--force_train\
	--anneal_beta\
	#--equalize\



# if contrast_treshold is greater than zero, brightness_treshold should be zero and vice versa
# if brightness treshold is zer0, contrast_treshold should be 11      
# these tresholds were only tested in Nguyen_march_2017
