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
	--data=Nguyen_march_2017\
	--texture=carpet\
	--z_dim_constrained=4\
        --with_mask\
        --with_prob\
        --save_preds\


