#!/bin/bash

#for fold in kutupalong_sept_2017 Deghale_Apr_2017 Kule_tirkidi_jun_2018 Kule_tirkidi_marc_2017 Minawao_feb_2017 Minawao_june_2016 Nguyen_march_2017 Tza_oct_2016 Zamzam_april_2022
#do 
#python ae_test.py\
#	--exp=ss_cvae\
#    	--model=ss_cvae\
#    	--num_epochs=100\
#    	--lr=1e-4\
#    	--img_size=256\
#    	--batch_size=16\
#    	--batch_size_test=8\
#    	--latent_img_size=32\
#    	--z_dim=256\
#    	--lamda=0.9\
#    	--nb_channels=4\
#    	--params_id=100\
#    	--dst_dir=D:/DATA/MVCamp/FEATURSPACE_all_anneal\
#    	--data_dir=D:/DATA/rawdata/camp\
#	--data=$fold\
#    	--dataset=camp\
#	--texture=carpet\
#    	--save_preds\
	#--with_mask\
	#--validate\

#mv D:/DATA/MVCamp/FEATURSPACE_all_anneal/predictions D:/DATA/MVCamp/FEATURSPACE_all_anneal/pred_$fold
#mv D:/DATA/MVCamp/FEATURSPACE_all_anneal/torch_features D:/DATA/MVCamp/FEATURSPACE_all_anneal/feat_$fold

#done


#Deghale_Apr_2017 Kule_tirkidi_jun_2018 Kule_tirkidi_marc_2017 Minawao_feb_2017 Minawao_june_2016 Nguyen_march_2017 Tza_oct_2016 Zamzam_april_2022

#for fold in Deghale_Apr_2017 Kule_tirkidi_jun_2018 Kule_tirkidi_marc_2017 kutupalong_sept_2017 Minawao_feb_2017 Minawao_june_2016 Nguyen_march_2017 Tza_oct_2016 Zamzam_april_2022
#do
python ae_test.py\
	--exp=dis_ssvae\
        --model=dis_ssvae\
        --num_epochs=25\
        --lr=1e-4\
        --img_size=256\
        --batch_size=16\
        --batch_size_test=8\
        --latent_img_size=32\
        --z_dim=36\
        --lamda=0.9\
        --nb_channels=4\
        --params_id=100\
        --dst_dir=D:/DATA/MVCamp_dis/outputs_34full_m4mm_mean_mask_m+\
	--data_dir=D:/DATA/rawdata/camp\
	--dataset=camp\
	--data=Minawao_june_2016\
	--texture=carpet\
	--z_dim_constrained=4\
        --with_mask\
        --with_prob\
        --save_preds\

#done

        #--validate\ Minawao_june_2016,Nguyen_march_2017,Tza_oct_2016,Minawao_feb_2017
	# Minawao_june_2016,Nguyen_march_2017,Tza_oct_2016

#done
