# SSL-VAE
This is the implementation of code used for [Self-Supervised Variational Autoencoder for Unsupervised Object Counting from Very-High-Resolution Satellite Imagery: Applications in Dwelling Extraction in FDP Settlement Areas](https://doi.org/10.1109/TGRS.2023.3345179) where the approach follows an anomaly detection approach with self-supervision. The approach is summarized in the workflow Figure below.

![workflow](ssvae_workflow.png)

## Usage

## USAGE 

To train a model
```
sh ae_train.sh
```

To test a model
```
sh ae_test.sh 
```

where specific parameters could be changed after opening the files
for feature space plot 
```
python automate_feature_space_plot.py --data_root ./root2data --save_dir ./root2save
```
