# train_file

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
import os
from datetime import datetime
from libs.data import get_loader, data_num, test_dataset
from libs.log import save_loss, save_lr, create_logger
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
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay
    return optimizer
def save_lr(save_dir, optimizer):
    update_lr_group = optimizer.param_groups[0]
    fh = open(save_dir, 'a')
    fh.write('decode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('\n')
    fh.close()
def save_loss(save_dir, whole_iter_num, epoch_total_loss, epoch_loss, epoch):
    fh = open(save_dir, 'a')
    epoch_total_loss = str(epoch_total_loss)
    epoch_loss = str(epoch_loss)
    fh.write('until_' + str(epoch) + '_run_iter_num' + str(whole_iter_num) + '\n')
    fh.write(str(epoch) + '_epoch_total_loss' + epoch_total_loss + '\n')
    fh.write(str(epoch) + '_epoch_loss' + epoch_loss + '\n')
    fh.write('\n')
    fh.close()
def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')        # 对应的类是torch.nn.BCEWithLogitsLoss
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()
def train(train_loader, model, optimizer, epoch):
    model.train()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts = pack
        images = Variable(images)
        gts = Variable(gts)
        images = images.to(device)
        gts = gts.to(device)
#-----------------------------------------------------------------------
        result = model(images)
        loss = structure_loss(result, gts)
#--------------------------------------------------------------------------
        loss.backward()
        clip_gradient(optimizer, config.clip)
        optimizer.step()
        #if i % 400 == 0 or i == total_step:
        logger.info('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                format(datetime.now(), epoch, config.epoch, i, total_step, loss.data))
    # 创建文件夹
    if not os.path.exists(config.weight_save_path):
        os.makedirs(config.weight_save_path)
    if (epoch+1) % 5 == 0:
        torch.save(model.state_dict(), config.weight_save_path +  config.model_name + '.%d' % epoch)
if __name__ == '__main__':
    import pprint
    model_name = config.model_name
    print(model_name)
    if model_name is None:
        model_name = os.path.abspath('').split('/')[-1]

    logger.info(config.model_name)
    logger.info(pprint.pformat(config))
    logger.info('''
    {}\n
    Starting training:
        Batch size: {}
        Learning rate: {}
        Train image size: {}
    '''.format(datetime.now(), config.batchsize, config.lr, config.imagesize))

    logger.info("=> building model")

    # Init model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = network()
    model = model.to(device)
    params = model.parameters()
    optimizer = torch.optim.Adam(params, config.lr)


    for epoch in tqdm(range(1, config.epoch)):

        logger.info("Starting epoch {}/{}.".format(epoch, config.epoch))

        optimizer = adjust_lr(optimizer, config.lr, epoch, config.decay_rate, config.decay_epoch)

        logger.info("epoch: {} ------ lr:{}".format(epoch, optimizer.param_groups[0]['lr']))

        datasets = os.listdir(config.train_img)

        for dataset in datasets:
            image_root1 = config.train_img + dataset + '/'
            gt_root1 = config.train_gt + dataset + '/'
            # load data
            train_loader = get_loader(image_root1, gt_root1, batchsize=config.batchsize, trainsize=config.imagesize)
            total_step = len(train_loader)
            train(train_loader, model, optimizer, epoch)
    logger.info('Epoch finished !!!')