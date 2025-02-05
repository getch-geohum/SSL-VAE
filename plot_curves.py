import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d



def plotOptimize():
    dicts = {}
    folds = [0,1,2,3,4,5,11]
    cols = ['b', 'g', 'r', 'c', 'm', 'k','y']
    for i in folds:
        auc = []
        for j in range(1, 101):
            fold = f'D:/DATA/Minawao_june_2016_0_{i}/torch_logs/{j}_AUC_ssmsummary.txt'
            with open(fold, 'r') as txt:
                read = txt.readlines()
                read = read[0].replace('\n', '').split(' ')
                auc.append(float(read[-1]))
        dicts[i] = auc
    plt.rcParams["figure.figsize"] = (12,12)    
    for k, c in enumerate(folds):
        if c == 11:
            plt.plot(gaussian_filter1d(dicts[c],sigma=1), linewidth=1.5, color = cols[k], label = r'no annealing $ \beta = 1 $')
        elif c == 0:
            plt.plot(gaussian_filter1d(dicts[c],sigma=1), linewidth=1.5, color = cols[k], label = r'anneal to $ \beta = 0.01 $')
        else:
            plt.plot(gaussian_filter1d(dicts[c],sigma=1), linewidth=1.5, color = cols[k], label = r'anneal to $ \beta = 0.{} $'.format(c))
        plt.hlines(y=0.9, xmin=-1, xmax=101, colors=None, linestyles='dotted')
        plt.hlines(y=0.92, xmin=-1, xmax=101, colors=None, linestyles='dotted')

        plt.xlabel('Epochs')
        plt.ylabel('Test performance(ROCAUC)')
        plt.legend()
        plt.savefig('/optimized_fig_smoth1.png', dpi=350, bbox_inches='tight')
#         plt.show()
        
if __name__ == "__main__":
    plotOptimize()
