#!/bin/bash

for fold in kutupalong_sept_2017 Deghale_Apr_2017 Kule_tirkidi_jun_2018 Kule_tirkidi_marc_2017 Minawao_feb_2017 Minawao_june_2016 Nguyen_march_2017 Tza_oct_2016 Zamzam_april_2022
do 
python ae_test.py\
	--exp=ssae\
    	--model=ssae\
    	--num_epochs=100\
    	--lr=1e-4\
    	--img_size=256\
    	--batch_size=16\
    	--batch_size_test=8\
    	--latent_img_size=32\
    	--z_dim=128\
    	--lamda=0.9\
    	--nb_channels=4\
    	--params_id=100\
    	--dst_dir=/home/getch/ssl/SSLAE_rev\
    	--data_dir=/home/getch/DATA/VAE/data\
	--data=$fold\
    	--dataset=camp\
	--texture=carpet\
	--delta=1\
    	--save_preds\
	--liu_vae\
	#--with_mask\
	#--validate\

#mv D:/DATA/MVCamp/FEATURSPACE_all_anneal/predictions D:/DATA/MVCamp/FEATURSPACE_all_anneal/pred_$fold
#mv D:/DATA/MVCamp/FEATURSPACE_all_anneal/torch_features D:/DATA/MVCamp/FEATURSPACE_all_anneal/feat_$fold

done


#Deghale_Apr_2017 Kule_tirkidi_jun_2018 Kule_tirkidi_marc_2017 Minawao_feb_2017 Minawao_june_2016 Nguyen_march_2017 Tza_oct_2016 Zamzam_april_2022

##for exp in dis_ssvae dis_ssvae_l dis_ssvae+l dis_ssvae+ll
#for exp in dis_ssvae dis_ssvae_l0_1r dis_ssvae_l0_1_ dis_ssvae_l0_1r dis_ssvae_l0_1_ 
#do
#	for fold in Deghale_Apr_2017 Kule_tirkidi_jun_2018 Kule_tirkidi_marc_2017 Minawao_feb_2017 Minawao_june_2016 Nguyen_march_2017 Tza_oct_2016 Zamzam_april_2022
#	do
#		python ae_test.py\
#			--exp=dis_ssvae\
#        		--model=dis_ssvae\
#        		--num_epochs=25\
#        		--lr=1e-4\
#        		--img_size=256\
#        		--batch_size=16\
#        		--batch_size_test=16\
#        		--latent_img_size=32\
#        		--z_dim=18\
#        		--lamda=0.9\
#        		--nb_channels=4\
#        		--params_id=100\
#        		--dst_dir=/home/getch/ssl/SSL_VAE/$exp\
#			--data_dir=/home/getch/DATA/VAE/data\
#			--dataset=camp\
#			--data=$fold\
#			--texture=carpet\
#			--z_dim_constrained=2\
#        		--with_mask\
#        		--with_prob\
#        		--save_preds\
#
#		done
#done

        #--validate\ Minawao_june_2016,Nguyen_march_2017,Tza_oct_2016,Minawao_feb_2017
	# Minawao_june_2016,Nguyen_march_2017,Tza_oct_2016

#done
