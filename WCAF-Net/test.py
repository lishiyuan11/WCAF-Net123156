# test_file
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from numpy import mean
import time
import datetime
import os
from scipy import misc
from libs.data import test_dataset
from libs.log import create_logger
from tqdm import tqdm
from importlib import import_module
import config

logger = create_logger('train')

try:
    network = getattr(import_module(config.model_name), config.network_name)     # from AUNet import AUNet
    print('import from config.')
    logger.info('import from config.')
except:
    print('import from py file.')
    logger.info('import from py file.')

model = network()
model.load_state_dict(torch.load(config.weight_load_path))
model.cuda()
model.eval()

start = time.time()
fps_list = []
logger = create_logger('test')
logger.info('model name:{}'.format(config.model_name))

for dataset in tqdm(config.test_datasets):
    dataset_img_path = config.test_path + '/' + dataset + '/img/'
    dataset_gt_path = config.test_path + '/' + dataset + '/gt/'
    img_classes = os.listdir(dataset_img_path)
    print(dataset)
    logger.info('dataset:{}'.format(dataset))

    for img_class in tqdm(img_classes):
        image_root = dataset_img_path + '/' + img_class + '/'
        gt_root = dataset_gt_path + '/' + img_class + '/'

        test_loader = test_dataset(image_root, gt_root,config.imagesize)
        time_list = []

        save_path = config.prediction_save_path + dataset + '/' + img_class + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)

            start_each = time.time()

            image = image.cuda()
            res = model(image)

            time_each = time.time() - start_each
            time_list.append(time_each)
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            cv2.imwrite(save_path+name, res*255)
            fps = 1 / mean(time_list)
            fps_list.append(fps)

end = time.time()
fps_mean = mean(fps_list)
logger.warning('fps_mean:{}'.format(fps_mean))
logger.warning("{}\nTotal Testing Time: {}".format(config.weight_load_path, str(datetime.timedelta(seconds=int(end - start)))))
