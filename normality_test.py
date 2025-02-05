from scipy.stats import ks_2samp
import statsmodels.api as sm
import numpy as np
import math
import matplotlib.pyplot as plt
from glob import glob
import argparse
import matplotlib
import os
#matplotlib.use("Agg")
np.random.seed(1)

def testNormality(x):
    '''Two sided test of normality for two independednt distributions, 
    x: is impirical distribution which a distribution of compressed latent space
    x2: computed below is a theoretical normal gaussian distribution with mean 0 and standard deviation of 1
    for implementation detailes on the test see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html and
    for further notes see  Hodges Jr, J. L. (1958). The significance probability of the Smirnov two-sample test. Arkiv för matematik, 3(5), 469-486. '''
    x2 = np.random.normal(0, 1, x.shape[0]) # gaussian normal distribution with eqaul size of the latent space for a specific dataset
    tst = ks_2samp(x,x2)
    return tst

def compareHistogram(root, out_root, z_dim=18,save=True):
    cons = {"18":['outputs_34full_vae_18','outputs_34full_m2mm'],
            "27":['outputs_34full_vae_27','outputs_34full_m3mm'],
            "36":['outputs_34full_vae_36','outputs_34full_m4mm']}

    subs = cons[str(z_dim)]
    names = ["Dagahaley-2017",
            "Kuletirkidi-2018",
            "Kuletirkidi-2017",
            "Minawao-2017",
            "Minawao-2016",
            "Nguenygiel-2017",
            "Nduta-2016",
            "Zamzam-2022",
            "Kutupalong-2017"]

    fig, ax = plt.subplots(1,9, figsize=(22, 4), sharey=True)
    for j, fold in enumerate(sorted(os.listdir(f'{root}/{subs[0]}/torch_features'))):
        files_a = glob(f'{root}/{subs[0]}/torch_features/{fold}/*.npy')
        files_b = glob(f'{root}/{subs[1]}/torch_features/{fold}/*.npy')

        ARS_a = [np.load(a) for a in files_a]
        ARS_b = [np.load(b) for b in files_b]

        ARS_a = np.concatenate(ARS_a).ravel()
        ARS_b = np.concatenate(ARS_b).ravel()

        mu_a =  np.round_(np.mean(ARS_a),3)
        std_a = np.round_(np.std(ARS_a),3)
        mu_b = np.round_(np.mean(ARS_b),3)
        std_b = np.round_(np.std(ARS_b),3)

        ax[j].hist(ARS_a, bins=2000, histtype = 'step', color='lime', label='Classical VAE')
        ax[j].hist(ARS_b, bins=2000, histtype = 'step', color='blue', label='VAE with latent space conditioning')
        ax[j].set_title(names[j])
        

        #print(f'mu1 {mu_a}, std1 {std_a}, mu2 {mu_b},std2 {std_b}')
        ax[j].text(0.45, 0.95,r'$\mu=${:0.3f}'.format(mu_a) + " " + r'$\sigma=${:0.3f}'.format(std_a), horizontalalignment='center', verticalalignment='center', transform=ax[j].transAxes, color='lime')
        ax[j].text(0.45, 0.9,r'$\mu=${:0.3f}'.format(mu_b) + " " + r'$\sigma=${:0.3f}'.format(std_b), horizontalalignment='center', verticalalignment='center', transform=ax[j].transAxes,color='blue')
    ax[0].set_ylabel('Frequency')
    fig.text(0.5, 0, 'Latent space logits', ha='center')
    #fig.text(0.04, 0.5, 'common Y', va='center', rotation='vertical')
    plt.subplots_adjust(wspace=0.001, hspace=0.001)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles,labels, loc='upper center', ncol=2, fontsize=10, bbox_to_anchor=(0.5,1.050))
    if save:
        plt.savefig(f'{out_root}/new_compare_{z_dim}.png', dpi=350, bbox_inches='tight')
    plt.show()


def jointHistogram(root, save=True, test_normality=True, out_root=None, name=None):
    names = ["Dagahaley-2017",
            "Kuletirkidi-2018",
            "Kuletirkidi-2017",
            "Minawao-2017",
            "Minawao-2016",
            "Nguenygiel-2017",
            "Nduta-2016",
            "Zamzam-2022",
            "Kutupalong-2017"]

    colors = ['blue', 'lime','darkgreen', 'red', 'cyan', 'yellow','black','fuchsia','olive']
    if save and out_root is None:
        raise ValueError('Please provide the path to save the plot')
    if test_normality:
        rep = open(f'{out_root}/{name}.txt', 'a+')
        rep.write('dataset,test_statistic, p-value\n')
    folds = sorted(os.listdir(root))
    fig, ax = plt.subplots(1,1,figsize=(15,15))
    for j,fold in enumerate(folds):
        files = glob(f'{root}/{fold}/*.npy')
        ARS = [np.load(a) for a in files]
        ARS = np.concatenate(ARS).ravel()      # treating each reduced latent space pixel as a single observation
        ax.hist(ARS,bins=2000, histtype = 'step', color=colors[j], label=names[j])
        if test_normality:
            test_rep = testNormality(ARS)
            rep.write(f'{names[j]},{test_rep.statistic},{test_rep.pvalue}\n')
    rep.close()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=10, bbox_to_anchor=(0.5,0.990))
    if save:
        plt.savefig(f'{out_root}/{name}.png', dpi=350, bbox_inches='tight')
    plt.show()
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="path for all feature spaces",type=str, required=True)
    parser.add_argument("--save",help="whether to save the plot", dest="save",action="store_true",)
    parser.set_defaults(save=False)
    parser.add_argument("--test_normality",help="whether to test normality", dest="test_normality",action="store_true",)
    parser.set_defaults(test_normality=False)
    parser.add_argument("--out_root", help="path to save the plots", type=str)
    parser.add_argument("--name", default='vae', type=str)
    parser.add_argument("--z_dim", default='18', type=int)
    parser.add_argument("--plot_type", default='compare', type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.plot_type != 'compare':
        jointHistogram(root=args.root,
                save=args.save,
                test_normality=args.test_normality,
                out_root=args.out_root,
                name=args.name)
    else:
        compareHistogram(root=args.root,
                out_root=args.out_root,
                z_dim=args.z_dim,
                save=args.save)
