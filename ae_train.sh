#!/bin/bash
python ae_train.py\
	--model=liu_vae\
	--exp=liu_vae\
	--num_epochs=100\
	--lr=0.0001\
	--img_size=256\
	--batch_size=16\
	--batch_size_test=16\
	--latent_img_size=32\
	--z_dim=128\
	--lamda=0.9\
	--nb_channels=4\
	--fake_dataset_size=400\
	--params_id=100\
	--dst_dir=/home/getch/ssl/LUVAE_rev/\
	--data_dir=/home/getch/DATA/VAE/data\
	--data=Deghale_Apr_2017,Kule_tirkidi_marc_2017,Minawao_feb_2017,Nguyen_march_2017,Zamzam_april_2022,Kule_tirkidi_jun_2018,kutupalong_sept_2017,Minawao_june_2016,Tza_oct_2016\
	--max_beta=1\
	--dataset=camp\
	--z_dim_constrained=28\
	--delta=1\
	--force_train\
	--with_mask\
	--with_prob\
	--liu_vae\
	
   
