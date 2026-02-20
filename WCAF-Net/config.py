import os

################# Training #################
model_name = 'HFINet'
network_name = 'HFINet'
epoch = 60
lr = 1e-4
batchsize = 10
imagesize = 256
clip = 0.5
decay_rate = 0.1
decay_epoch = 50
num_worker = 1
pretraind_pth_path = './pth/backbone/pvt_v2_b2.pth'
train_img = '../train/CoCOD/image/'
train_gt = '../train/CoCOD/groundtruth/'
weight_save_path = './pth/' + model_name + '/'              # 权重文件保存路径

################# Test #################
test_datasets = ['CoCOD', 'CAMO', 'CHAMELEON', 'COD10K']     # 第一个作为训练时的评估数据
test_path = "../test/"
prediction_save_path = './prediction/' + model_name + '/'
weight_load_path = weight_save_path + model_name + '.59'     # 加载权重文件
