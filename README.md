# SSL-VAE
This is the implemntation of self-supervised auto and variational encoding for unsupervised and self-supervised dwelling localization and counting in frocibly displaced population(FDP) sites.
As we have discussed the wea anomal creation is datset centext specific and if you want to try in your side, here are some tresholds of which you can change in ae_train.sh file

And this is a working  project still under progress. It will be public soon its completion.!

1. datset: Minawao_june_2016    **_func=NDVI, ndvi_treshold=0.1_**
2. datset: Minawao_feb_2017     **_func=NDVI, ndvi_treshold=0.3_**
3. dataset: Deghale_Apr_2017    **_func=NDVI, ndvi_treshold=0.1_**
4. dataset: Zamzam_april_2022   **_func=NDVI, ndvi_treshold=0.2_**
5. dataset:Nguyen_march_2017    **_func=grayIntensity, intensity_treshol=120, brightness_treshold=6.5 contrast_teshold=0_**
6. dataset: Tza_oct_2016        **_func=grayIntensity, intensity_treshol=120, brightness_treshold=7 contrast_teshold=0_**
7. dataset: Kutuplong_dec_2017  **_func=channelIntensity, intensity_treshol=150, brightness_treshold=0 contrast_teshold=0_**
8. dataset: Kule_tirkidi_jun_2018  **_func=grayIntensity, intensity_treshol=140, brightness_treshold=0 contrast_teshold=0 equalize_**
9. dataset: Kule_tirkidi_marc_2017  **_func=grayIntensity, intensity_treshol=145, brightness_treshold=0 contrast_teshold=0 equalize_**

# Contributing
Install pre-commit, run `pre-commit install` once in your repository to get the hooks from `pre-commit-config.yaml` triggered at each commit.
