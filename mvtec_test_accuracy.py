import numpy as np
import os
from skimage.io import imread
import cv2
import argparse
from glob import glob
from sklearn.metrics import confusion_matrix


def computMetrics(ref, pred, e=1e-6):
    assert ref.shape == pred.shape, 'shapes of predicted and refrence images are not the same'
    ref = ref.ravel()
    pred = pred.ravel()
    
    c_mat = confusion_matrix(ref, pred)   # confusion matrix
    tn, fp, fn, tp = c_mat.ravel()
    
    tpr = tp/(tp + fn + e)
    tnr = tn/(tn + fp + e)
    mean = (tpr + tnr)/2
    f1 = 2*tp /(2*tp + fp + fn + e)
    
    return {'tpr':tpr, 'tnr':tnr, 'mean': mean, 'f1':f1}

def MaxValue(img):
    return img.max()

def pQuantile(img, p):
    q = np.percentile(img, p) 
    return q

def kSigma(img, k):
    mean = img.mean()
    std = img.std()
    t = mean + k*std
    return t

class Findtreshold:
    def __init__(self, root, method='max', p=99, k=1):
        self.root = root
        self.method = method
        self.p = p
        self.k = k
        
    def computeTreshold(self):
        files = sorted([self.root + f'/{file}' for file in os.listdir(self.root) if '_amap' in file])
        self.big_array = np.zeros((512,512,len(files)))
        for j, file in enumerate(files):
            arr = imread(file)
            gray = cv2.cvtColor(arr,cv2.COLOR_BGR2GRAY)
            self.big_array[:,:,j] = gray
            
        if self.method == 'max':
            self.treshold = MaxValue(self.big_array)
        elif self.method == 'p-quantile':
            self.treshold = pQuantile(self.big_array, p=self.p)
        elif self.method == 'k-sigma':
            self.treshold = kSigma(self.big_array, k=self.k)
        else:
            raise ValueError(f'Provided method: {self.method} is not known')
        return self.treshold


def generateAccuracMetrix(main_root, texture, method='max', p=99, k=1, save_segment=False):
    """ method: methode to compute the matrix
        p: percentile above whih the pixels are considered anomalies
        k: the standard deviation units for k-sigma values
    """
    normal_input_root = main_root + f'/{texture}/predictions_good'
    input_root = main_root + f'/{texture}/predictions'
    out_root = main_root + f'/{texture}/torch_logs'
    print('computing treshold')
    finder = Findtreshold(root=normal_input_root, method = method, p=p, k=k)
    treshold = finder.computeTreshold()
    print(f'Treshold defined: {treshold}')
    refs = sorted(glob(f'{input_root}/*_gt.png'))
    preds = sorted(glob(f'{input_root}/*_amap.png'))
    
    assert len(refs) == len(preds), 'predicted and reference datasets are not the same'
    
    zips = list(zip(preds, refs))
    print(f'Got {len(zips)} image pairs for accuracy metrics')
    TPR = []
    TNR = []
    MEAN = []
    F1 = []
    print('Running segmentation metrics ...')
    for j, data in enumerate(zips):
        ref_r = imread(data[1])
        pred_r = cv2.cvtColor(imread(data[0]),cv2.COLOR_BGR2GRAY) # to make single channel 8 bits image
        mask = np.where(pred_r>=treshold, 1, 0)
        metrics = computMetrics(ref=ref_r, pred=mask)
        TPR.append(metrics['tpr'])
        TNR.append(metrics['tnr'])
        MEAN.append(metrics['mean'])
        F1.append(metrics['f1'])
        print(metrics['tpr'], metrics['tnr'], metrics['mean'], metrics['f1'])
        if save_segment:
            img_to_save = Image.fromarray(mask.astype(np.uint8))
            km = os.path.split(data[0])[1][:-4]
            img_to_save.save(input_root + f'/{km}_seg.png')
            
    mean_tpr = np.nanmean(TPR)
    mean_tnr = np.nanmean(TNR)
    mean_mean = np.nanmean(MEAN)
    mean_f1 = np.nanmean(F1)
    
    name = f'{out_root}/segment_metrics.txt'
    
    with open(name, 'a+') as txt:
        txt.write('++======================++\n')
        txt.write(f'=== Method: {method}, k: {k}, p: {p} ====\n')
        txt.write(f'mean TPR: {mean_tpr}\n')
        txt.write(f'mean TNR: {mean_tnr}\n')
        txt.write(f'mean of TPR and TNR: {mean_mean}\n')
        txt.write(f'mean F-score: {mean_f1}\n')
        txt.write('++======================++\n')
    print(f'Prediction done for {texture}')


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help='Root path containing all texture predcitions')
    parser.add_argument("--texture", help="The texture calss of mvtec dataset", default="carpet", type=str)
    parser.add_argument("--p", default=98, type=int)
    parser.add_argument("--k", default=2, type=int)
    parser.add_argument("--method", default=max, type=str)
    parser.add_argument("--save_segment", help='whether to save the segmentation results or not', dest='save_segment', action='store_true')
    parser.set_defaults(save_segment=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    generateAccuracMetrix(main_root=args.root,
                          texture=args.texture,
                          method=args.method,
                          p=args.p,
                          k=args.k,
                          save_segment=args.save_segment)
