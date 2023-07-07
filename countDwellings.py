from torch import BoolTensor, IntTensor, Tensor
import numpy as np
from skimage.draw import polygon
from shapely.geometry import Polygon
from shapely.validation import make_valid
from skimage.io import imread, imsave
from skimage import measure
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint
import argparse
import os
from glob import glob
from sklearn.mixture import GaussianMixture

def MASK2PASCAL(MASK, min_area=4, with_score=False):
    contours = measure.find_contours(MASK, 0.5)
    boxes = []
    label = []
    masks = []
    scores = [] # basically this is as a paceholder for torch metrics

    if len(contours)>=1:
        for cont in contours:
            for i in range(len(cont)):
                row, col = cont[i]
                cont[i] = (col, row)
            if len(cont)<4: # invalid geometries
                continue
            poly = Polygon(cont)
            if poly.is_empty:
                continue
            if poly.geom_type == 'MultiPolygon':
                for spoly in poly:
                    if spoly.is_empty:
                        continue
                    if not spoly.is_valid:
                        continue
                    min_x, min_y, max_x, max_y = spoly.bounds
                    bboox = [min_x, min_y, max_x, max_y]
                    area = spoly.area
                    img = np.zeros(MASK.shape)
                    rr, cc = polygon(cont[:, 0], cont[:, 1], MASK.shape)
                    img[rr, cc] = 1
                    if area > min_area:
                        boxes.append(bboox)
                        label.append(0)
                        masks.append(img)
                        scores.append(0.9)

            else:
                if not poly.is_valid:
                    continue
                min_x, min_y, max_x, max_y = poly.bounds
                bboox = [min_x, min_y, max_x, max_y]
                area = poly.area
                img = np.zeros(MASK.shape)
                rr, cc = polygon(cont[:, 0], cont[:, 1], MASK.shape)
                img[rr, cc] = 1
                if area > min_area:
                    boxes.append(bboox)
                    label.append(0)
                    masks.append(img)
                    scores.append(0.9)

    preds = {
            "boxes": Tensor(boxes),
            "labels": IntTensor(label),
            "masks": BoolTensor(masks)
            }
    if with_score:
        preds["scores"] = Tensor(scores)

    return preds, len(boxes) # with object count


class Segmenter:
    def __init__(self, root=None, num_class=2, min_area=6, score='amap', iou_type='segm'):
        self.root = root
        self.score = score
        self.num_class = num_class
        self.min_area = min_area
        self.score_maps = sorted(glob(f'{self.root}/*_{score}.png'))
        self.ground_truth = sorted(glob(f'{self.root}/*_gt.png'))

        assert len(self.score_maps) == len(self.ground_truth), f'score maps and ground truth {len(self.score_maps)} and {len(self.ground_truth)} respectively is not equal'

        self.model = GaussianMixture(n_components=self.num_class, covariance_type='full',random_state=0, init_params='kmeans')
        self.metric = MeanAveragePrecision(iou_type=iou_type)

    def fitModel(self):
        print('fitting the global model...')
        images = np.vstack([imread(im).reshape(-1,3) for im in self.score_maps])
        self.model.fit(images)

    def segmentCount(self):
        print('Running segmentation...')
        files = list(zip(self.score_maps, self.ground_truth))
        REF = []
        PRE = []
        RC = []
        PC = []

        for pair in files:
            A = imread(pair[0])
            G = imread(pair[1])

            p_mask = self.model.predict(A.reshape(-1,3))
            p_mask = p_mask.reshape(A.shape[0], A.shape[1])
            count = np.unique(p_mask, return_counts=True)
            vals = [count[1][0]/count[1].sum(), count[1][1]/count[1].sum()]

            if vals[1] > 0.7:  # for images with emty background, just to control unnecessary confusion
                pass
            else:
                rf = MASK2PASCAL(MASK=G, min_area=self.min_area, with_score=False)
                pr = MASK2PASCAL(MASK=p_mask, min_area=self.min_area, with_score=True)

                REF.append(rf[0])
                PRE.append(pr[0])
                RC.append(rf[1])
                PC.append(pr[1])
                print(f'Ref --> {rf[1]} Pred --> {pr[1]}')

        self.preds = PRE
        self.refs = REF
        self.rc = RC
        self.pc = PC

    def computeMetrics(self):
        print('Computing objective metrics...')
        self.metric.update(self.preds, self.refs)
        self.results = self.metric.compute()

    def summurizeReport(self,out_root=None):
        print('Summurizing the report...')
        if not os.path.exists(out_root):
            os.makedirs(out_root, exist_ok=True)

        np.save(f'{out_root}/{self.score}_reff_count.npy', np.array(self.rc))
        np.save(f'{out_root}/{self.score}_pred_count.npy', np.array(self.pc))

        keys = ['map', 'map_50', 'map_75', 'map_small', 'map_medium', 'map_large', 'ref_count', 'pred_count']
        with open(f'{out_root}/{self.score}_metric.txt', 'a') as txt:
            txt.write(f'===================== \n')
            for key in keys:
                if key in ['ref_count']:  # reference count
                    txt.write(f'{key}, {np.nansum(self.rc)} \n')
                elif key in ['pred_count']: # predicted count
                    txt.write(f'{key}, {np.nansum(self.pc)} \n')
                else:
                    txt.write(f'{key}, {self.results[key].item()} \n') # mean average precision



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="root folder where files exist", type=str, required=False)
    parser.add_argument("--out_root", help="root folder where files will be saved", type=str, required=False)
    parser.add_argument("--num_class", type=int, default=2)
    parser.add_argument("--min_area",help="The minimum area to define objectness", default=6, type=int)
    parser.add_argument("--score",help="Anomaly score either amap, mads or mads_copy",type=str)
    parser.add_argument("--iou_type",help="iou measurement type either segm or bbox",type=str, default="segm")
    return parser.parse_args()




if __name__ == "__main__":
    args = parse_args()
    for folder in os.listdir(f'{args.root}/predictions'):
        print(f'Predicting for folder: {folder}')
        root = f'{args.root}/predictions/{folder}'
        out_root = f'{args.out_root}/{folder}'
        #if folder in ['Tza_oct_2016']:
        segmenter = Segmenter(root=root,
                              num_class=args.num_class,
                              min_area=args.min_area,
                              score=args.score,
                              iou_type=args.iou_type)
        segmenter.fitModel()
        segmenter.segmentCount()
        segmenter.computeMetrics()
        segmenter.summurizeReport(out_root=out_root)


