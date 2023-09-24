import os
import cv2
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import utils_colour as utils_colour

from detectron2.engine import default_argument_parser
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

height, width = 480, 640


def load_img(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height)) / 255
    return img


def predict(input_img, predictor):
    sem_seg = predictor(input_img)['sem_seg']
    sem_seg = F.softmax(sem_seg, dim=0)
    # for i in range(41):
    #     if i in [0, 1, 2]:
    #         pass
    #     elif i == 8: # regard door as wall
    #         sem_seg[1] += sem_seg[i]
    #     elif i == 30: # regard white board as wall
    #         sem_seg[1] += sem_seg[i]
    #     elif i == 20: # regard floor mat as floor
    #         sem_seg[2] += sem_seg[i]
    #     else:
    #         sem_seg[0] += sem_seg[i]
    # sem_seg = sem_seg[[0, 1, 2]]
    score, sem_seg = sem_seg.max(dim=0)
    sem_seg = sem_seg.cpu().numpy()
    return sem_seg

def mapping_nyu3(manhattan=False):
    mapping = {}
    for i in range(41):
        if i in [0, 1, 2]:
            mapping[i]=i
        else:
            mapping[i]=0
        if manhattan:
            if i==8: # regard door as wall
                mapping[i]=1
            elif i == 30: # regard white board as wall
                mapping[i]=1
            elif i == 20: # regard floor mat as floor
                mapping[i]=2
    return mapping

def mapping_nyu40(manhattan=False):
    mapping = {}
    for i in range(41):
        mapping[i]=i
        if manhattan:
            if i==8: # regard door as wall
                mapping[i]=1
            elif i == 30: # regard white board as wall
                mapping[i]=1
            elif i == 20: # regard floor mat as floor
                mapping[i]=2
    return mapping

def main(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    cfg.merge_from_file('./docs/semantic_segmentation/configs/deeplabv3plus.yaml')
    cfg.MODEL.WEIGHTS = '../sem_model.pth'
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 41
    cfg.MODEL.PIXEL_MEAN = [0.5, 0.5, 0.5]
    cfg.MODEL.PIXEL_STD = [0.5, 0.5, 0.5]
    predictor = DefaultPredictor(cfg)
    scannet_dir='/home/du/Proj/NeuRIS/Data/dataset/indoor'
    for scene_name in [
        'scene0146_00','scene0086_00','scene0686_00'  # TODO: modify this to your path
    ]:
        img_path = os.path.join(scannet_dir,f'{scene_name}/image/')
        semantic_path = os.path.join(scannet_dir,f'{scene_name}/semantic_deeplab/')
        semantic_vis= os.path.join(scannet_dir,f'{scene_name}/semantic_deeplab_vis/')
        semantic_vis_40 = os.path.join(scannet_dir,f'{scene_name}/semantic_deeplab_40_vis/')
        semantic_vis_3 = os.path.join(scannet_dir,f'{scene_name}/semantic_deeplab_3_vis/')
        os.makedirs(semantic_path, exist_ok=True)
        os.makedirs(semantic_vis, exist_ok=True)
        os.makedirs(semantic_vis_40, exist_ok=True)
        os.makedirs(semantic_vis_3, exist_ok=True)

        imgs = os.listdir(img_path)
        imgs = [(os.path.splitext(frame)[0]) for frame in imgs]
        imgs =  sorted(imgs)
        for img_filename in tqdm(imgs):
            img = load_img(f'{img_path}/{img_filename}.png')
            sem_seg = predict(img, predictor)
            # sem_seg = (sem_seg * 80).astype(np.uint8)
            sem_seg = (sem_seg).astype(np.uint8)
            sem_seg = cv2.resize(sem_seg, (width, height), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(f'{semantic_path}/{int(img_filename)}.png', sem_seg)
            
            colour_map_np = utils_colour.nyu40_colour_code
            sem_vis = colour_map_np[sem_seg]
            cv2.imwrite(f'{semantic_vis}/{int(img_filename)}.png', sem_vis[...,::-1])
            # 可视化40类语义
            sem_seg_40=sem_seg.copy()
            mapping_40=mapping_nyu40(manhattan=True)
            for deeplab_id, output_id in mapping_40.items():
                sem_seg_40[sem_seg==deeplab_id] = output_id
                sem_seg_40=np.array(sem_seg_40)
            vis_40_label = colour_map_np[sem_seg_40]
            cv2.imwrite(f'{semantic_vis_40}/{int(img_filename)}.png', vis_40_label[...,::-1])
            # 可视化3类语义
            sem_seg_3=sem_seg.copy()
            mapping_3=mapping_nyu3(manhattan=True)
            for deeplab_id, output_id in mapping_3.items():
                sem_seg_3[sem_seg==deeplab_id] = output_id
                sem_seg_3=np.array(sem_seg_3)
            vis_3_label = colour_map_np[sem_seg_3]
            cv2.imwrite(f'{semantic_vis_3}/{int(img_filename)}.png', vis_3_label[...,::-1])

if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)
