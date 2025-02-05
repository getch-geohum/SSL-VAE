#!/bin/bash
for j in 0 11
do
	echo processing for value $j
	for i in {1..100}
	do
		echo "computing metrics for epoch $i"
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
    			--params_id=$i\
    			--dst_dir=/path2save$j\
    			--data_dir=/path2data\
    			--with_mask\
    			--with_prob\
    			#--save_preds\
	done
done


