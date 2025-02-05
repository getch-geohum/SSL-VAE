#!/bin/bash
#Deghale_Apr_2017 Kule_tirkidi_jun_2018 Nguyen_march_2017 Zamzam_april_2022
for dir in Kule_tirkidi_marc_2017 Minawao_feb_2017 Minawao_june_2016 Tza_oct_2016 Deghale_Apr_2017 Kule_tirkidi_jun_2018 Nguyen_march_2017 Zamzam_april_2022
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
    		--z_dim=256\
    		--lamda=0.9\
    		--nb_channels=4\
    		--params_id=100\
    		--dst_dir=/path2save/$dir\
    		--data_dir=/path2data/$dir\
    		--with_mask\
    		--with_prob\
    		--save_preds\
		#--equalize
done
