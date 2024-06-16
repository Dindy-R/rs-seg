'''
@Description: 推理
'''
import os
import sys
import cv2
import torch
import argparse
import ttach as tta
import numpy as np
import os.path as osp
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets.ImgMaskDataset import get_val_transform
from datasets.ImgDataset import ImageDataset
from models import create_model
from utils import load_config, sementic_splash
from utils.pred import RSImagePredictManager, WeightedPredictManager

# to solve the problem of 'ERROR 1: PROJ: pj_obj_create: Open of /opt/conda/share/proj failed'
# os.environ['PROJ_LIB'] = '/opt/conda/share/proj'
# os.environ['PROJ_LIB'] = r'C:\Users\AI\anaconda3\envs\torch17\Library\share\proj'


<<<<<<< HEAD
class Segmenter(object):
    def __init__(self, cfg_name, weight):
        cfg = load_config(cfg_name, 'configs')
        self.cfgN = cfg['network']
        self.cfgN['pretrained'] = None
        self.cfgD = cfg['dataset']
        self.cfgI = cfg['infer']
        self.n_class = len(self.cfgD['cls_info'])
        self.transform = get_val_transform()
        log_dir = osp.join(cfg['run_dir'], cfg_name)
        #  check weight file
        ckpt_path = osp.join(log_dir, 'ckpt', weight)
        if not osp.exists(ckpt_path):
            print(f'file {ckpt_path} not found.')
            return
        self.model = create_model(cfg=self.cfgN).cuda()
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        # summary(model, input_size=(in_channel, in_height, in_width))
        if self.cfgI['tta']:
            tta_transforms = tta.aliases.d4_transform()
            # tta_transforms = tta.Compose(
            #     [
            #         tta.HorizontalFlip(),
            #         tta.VerticalFlip(),
            #         tta.Rotate90(angles=[0, 180]),
            #         tta.Scale(scales=[1, 1.5, 2]),
            #     ]
            # )
            self.model = tta.SegmentationTTAWrapper(self.model, tta_transforms, merge_mode='mean')

    @torch.no_grad()
    def predict(self, rgb_img):
        input = self.transform(image=rgb_img)['image']
        input = input.unsqueeze(0).cuda()
        output = self.model(input)    # (n, c, h, w)
        output = torch.softmax(output, dim=1)
        output = output.squeeze(0).cpu().numpy()
        output = np.argmax(output, axis=0).astype(np.uint8)  # (h, w)
        return output

    def predict_folder(self, in_dir, out_dir):
        # per image
        os.makedirs(out_dir, exist_ok=True)
        files = [f for f in os.listdir(in_dir) if f.endswith('.tif')]
        tbar = tqdm(files)
        tbar.set_description('Inference')
        for fname in tbar:
            image = cv2.imread(os.path.join(in_dir, fname))
            rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pred_mask = self.predict(rgb_img)
            out_path = os.path.join(out_dir, fname[:-4] + '.png')
            cv2.imwrite(out_path, pred_mask.astype(np.uint8))
=======
def pred_large_RSimagery_2level_windowed(
        model,
        img_path,
        out_path,
        l1_win_sz,
        l1_overlap,
        l2_win_sz,
        l2_overlap,
        batch_sz,
        transform,
        n_class,
        TTA=False):

    assert TTA==False   # not implement
    rs_mng = RSImageSlideWindowManager(
        in_raster=img_path,
        out_raster=out_path,
        out_bands=1,
        window_sz=l1_win_sz,
        overlap=l1_overlap)
    tbar = tqdm(range(len(rs_mng)))
    tbar.set_description(os.path.basename(img_path))
    for i in tbar:
        imdata_chw = rs_mng.get_next().copy()
        im_loader = DataLoader(
            dataset=ImageDataset(
                im_data=imdata_chw,
                tile_size=l2_win_sz,
                overlap=l2_overlap,
                transform=transform,
                channel_first=True),
            batch_size=batch_sz,
            shuffle=False,
            num_workers=batch_sz,
            drop_last=False)
        pred_mng = WeightedPredictManager(
            map_height=imdata_chw.shape[1],
            map_width=imdata_chw.shape[2],
            map_channel=n_class,
            patch_height=l2_win_sz,
            patch_width=l2_win_sz)
        with torch.no_grad():
            for batch_data in im_loader:
                patch, windows = batch_data['image'], batch_data['window']
                patch = patch.cuda()
                output = model(patch)
                # output = F.interpolate(output, size=(patch_size, patch_size), mode='bilinear')  # -> (n, c, h, w)
                probs = torch.softmax(output, dim=1).cpu().numpy()
                pred_mng.update(probs, windows.numpy())
>>>>>>> parent of c0d9aba (add sgementer.py)

            # colors_bgr = [[0, 0, 0], [255, 0, 0]]
            res = sementic_splash(image, pred_mask, n_label=self.n_class, alpha=1.0, beta=0.5)
            res_name = out_path.replace('.png', '_color.jpg')
            cv2.imwrite(res_name, res)

    def predict_large_imagery(self, in_path, out_path):
        rs_mng = RSImagePredictManager(
            in_raster=in_path,
            out_raster=out_path,
            window_sz=self.cfgI['l1_win_sz'],
            net_sz=self.cfgI['l2_win_sz'],
            overlap=self.cfgI['l1_overlap'])

        tbar = tqdm(range(len(rs_mng)))
        tbar.set_description(os.path.basename(in_path))
        for _ in tbar:
            imdata_chw, _ = rs_mng.get_next()
            im_loader = DataLoader(
                dataset=ImageDataset(
                    im_data=imdata_chw,
                    tile_size=self.cfgI['l2_win_sz'],
                    overlap=self.cfgI['l2_overlap'],
                    transform=self.transform,
                    channel_first=True),
                batch_size=self.cfgI['batch_size'],
                shuffle=False,
                num_workers=2,
                drop_last=False)
            pred_mng = WeightedPredictManager(
                map_height=imdata_chw.shape[1],
                map_width=imdata_chw.shape[2],
                map_channel=self.n_class,
                patch_height=self.cfgI['l2_win_sz'],
                patch_width=self.cfgI['l2_win_sz'])
            with torch.no_grad():
                for batch_data in im_loader:
                    patch, y1y2x1x2 = batch_data['image'], batch_data['window']
                    patch = patch.cuda()
                    output = self.model(patch)
                    # output = F.interpolate(output, size=(patch_size, patch_size), mode='bilinear')  # -> (n, c, h, w)
                    probs = torch.softmax(output, dim=1).cpu().numpy()
                    pred_mng.update(probs, y1y2x1x2.numpy())

            pred_mask, _ = pred_mng.get_result()
            # post processing

            rs_mng.update(pred_mask[None, :, :])
        rs_mng.close()

    def predict_large_imagery_folder(self, in_dir, out_dir):
        files = [f for f in os.listdir(in_dir) if f.endswith('.tif')]
        tbar = tqdm(files)
        tbar.set_description('Inference Large Imagery')
        for fname in tbar:
            img_path = os.path.join(in_dir, fname)
            out_path = os.path.join(out_dir, fname[:-4] + '.tif')
            self.predict_large_imagery(img_path, out_path)


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-c', '--config', type=str, required=True, help='Name of the config file.')
    argparser.add_argument('-w', '--weight', default='best.pt', type=str, help='Choice a weight file.')
    argparser.add_argument('-i', '--input', required=True, type=str)
    argparser.add_argument('-o', '--output', required=True, type=str)
    argparser.add_argument('-g', '--gpus', default='0', type=str, help='gpus')
    return argparser.parse_args()


<<<<<<< HEAD
if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    segmenter = Segmenter(args.config, args.weight)
    # segmenter.predict_folder(args.input, args.output)
    segmenter.predict_large_imagery_folder(args.input, args.output)
=======
    I = CFG['inference_params']
    ckpt            = os.path.join(CFG['run_dir'], CFG['run_name'], "ckpt", I['ckpt_name'])
    input_dir       = I['in_dir']
    base_dir        = os.path.join(CFG['run_dir'], CFG['run_name'], I['out_dir'])
    res_dir         = os.path.join(base_dir, 'results')
    l1_win_sz       = I['l1_win_sz']
    l1_overlap      = I['l1_overlap']
    l2_win_sz       = I['l2_win_sz']
    l2_overlap      = I['l2_overlap']
    batch_size      = I['batch_size']
    TTA             = I['tta']
    draw_mask       = I['draw']
    evaluate        = I['evaluate']
>>>>>>> parent of c0d9aba (add sgementer.py)


<<<<<<< HEAD
=======
    # network
    model = create_model(type=nn_type,
                         arch=arch,
                         encoder=encoder,
                         in_channel=in_channel,
                         out_channel=out_channel,
                         pretrained=pretrained).cuda()
    # model = torch.nn.DataParallel(model)
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    # summary(model, input_size=(in_channel, in_height, in_width))

    transform = get_val_transform(l2_win_sz)
    img_set = [f for f in os.listdir(input_dir) if f.endswith('.tif')]

    # Inference
    tbar = tqdm(img_set)
    tbar.set_description('Inference')
    for fid in tbar:
        # tbar.set_postfix_str(fid)
        img_path = os.path.join(input_dir, fid)
        out_path = os.path.join(res_dir, fid[:-4] + '.tif')
        pred_large_RSimagery_2level_windowed(
            model,
            img_path=img_path,
            out_path=out_path,
            l1_win_sz=l1_win_sz,
            l1_overlap=l1_overlap,
            l2_win_sz=l2_win_sz,
            l2_overlap=l2_overlap,
            batch_sz=batch_size,
            transform=transform,
            n_class=n_class,
            TTA=TTA)

'''
    # Evaluate
    if not evaluate: return
    txt = []
    cls_names = list(class_info.keys())
    M = Metric(n_class)
    tbar = tqdm(img_set)
    tbar.set_description('Evaluate')
    for fid in tbar:
        # tbar.set_postfix_str(fid)
        pred_path = os.path.join(res_dir, fid)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        label_path = os.path.join(input_dir.replace('images', 'labels'), fid)
        if os.path.isfile(label_path):
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            # label = (label > 0).astype(np.uint8)
        else:
            raise FileNotFoundError(f'file [{label_path}] not found.')
        M.add_batch(pred, label)

    scores = M.evaluate()
    mIoU = scores["mean_iou"]
    IoUs = scores["class_iou"]
    Ps = scores["class_precision"]
    Rs = scores["class_recall"]
    confusion_matrix = M.get_confusion_matrix()
    _output = '> mIoU: {:.4f}'.format(mIoU)
    print(_output)
    txt.append(_output + '\n')
    for i in range(n_class):
        _output = '{:<20}| IoU: {:<10.4f} P: {:<.4f} R: {:<.4f}'.format(cls_names[i], IoUs[i], Ps[i], Rs[i])
        print(_output)
        txt.append(_output + '\n')

    # write txt
    with open(os.path.join(base_dir, 'score.txt'), 'w', newline='') as f:
        f.writelines(txt)

    # confusion matrix to csv
    mcm_csv = pd.DataFrame(confusion_matrix, index=cls_names, columns=cls_names)
    mcm_csv.to_csv(os.path.join(base_dir, 'confusion_matrix.csv'))
'''

'''
@torch.no_grad()
def predict_in_large_RSimagery(model, img_path, patch_size, overlap, batch_size, transform, n_class, TTA=False):

    assert batch_size == 1
    assert TTA == False
    t0 = time.time()
    raster = gdal.Open(img_path)
    img_w = raster.RasterXSize
    img_h = raster.RasterYSize

    predmng = PredictManager(img_h, img_w, n_class, patch_size, patch_size)
    test_loader = raster_Generater(raster, patch_size, overlap, transform)
    for patch_data in tqdm(test_loader):
        patch, box = patch_data['image'], patch_data['box']
        patch = patch.cuda()
        output = model(patch)
        output = F.interpolate(output, size=(patch_size, patch_size), mode='bilinear') # -> (n, c, h, w)
        prob = torch.softmax(output, dim=1).cpu().numpy()

        y1, y2, x1, x2 = box
        predmng.update(prob.squeeze(0), yoff=y1, xoff=x1)

    whole_map = predmng.get_result()

    return whole_map, time.time() - t0
'''


'''

def predict_in_tile_image(model, image, transform, n_class, TTA=False):

    def _predict(img):
        t0 = time.time()
        img = transform(image=img)['image']
        img = img.unsqueeze(0)
        with torch.no_grad():
            img = img.cuda()
            output = model(img)
        output = output.permute(0, 2, 3, 1)  # -> (n, h, w, c)
        output = torch.softmax(output, dim=3)
        output = output.squeeze().cpu().numpy()
        return output, time.time() - t0

    if TTA:
        img_h, img_w, _ = image.shape
        prob_maps = np.zeros((img_h, img_w, n_class), dtype=np.float32)
        total_time = 0

        _map, _time = _predict(image)
        prob_maps += _map
        total_time += _time

        img_fr = np.fliplr(image)
        _map, _time = _predict(img_fr)
        _map = np.fliplr(_map)
        prob_maps += _map
        total_time += _time

        img_fv = np.flipud(image)
        _map, _time = _predict(img_fv)
        _map = np.flipud(_map)
        prob_maps += _map
        total_time += _time

        img_fr_fv = np.flipud(img_fr)
        _map, _time = _predict(img_fr_fv)
        _map = np.flipud(_map)
        _map = np.fliplr(_map)
        prob_maps += _map
        total_time += _time

        prob_maps = prob_maps / 4
    else:
        prob_maps, total_time = _predict(image)

    return prob_maps, total_time

'''


>>>>>>> parent of c0d9aba (add sgementer.py)
