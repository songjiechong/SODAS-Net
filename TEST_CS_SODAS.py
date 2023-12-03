import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
import glob
from time import time
import math
from torch.nn import init
import copy
import cv2
from skimage.measure import compare_ssim as ssim
from argparse import ArgumentParser
import pandas as pd

parser = ArgumentParser(description='ISTA-Net-plus')

parser.add_argument('--layer_num', type=int, default=25, help='phase number of ISTA-Net-plus')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--cs_ratio', type=int, default=30, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--noise', type=float, default=0, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--patch_size', type=int, default=99)
parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='Dataset', help='training or test data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')
parser.add_argument('--test_name', type=str, default='Set11', help='name of test set')
parser.add_argument('--algo_name', type=str, default='SODASNet', help='log directory')

args = parser.parse_args()

learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list
test_name = args.test_name
noise = args.noise

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ratio_dict = {1: 10, 4: 43, 10: 109, 25: 272, 30: 327, 40: 436, 50: 545}

n_input = ratio_dict[cs_ratio]
n_output = 1089
nrtrain = 88912
batch_size = 64


# Load CS Sampling Matrix: phi
Phi_data_Name = './%s/phi_0_%d_1089.mat' % (args.matrix_dir, cs_ratio)
Phi_data = sio.loadmat(Phi_data_Name)
Phi_input = Phi_data['phi']

# Define ISTA-Net-plus Block
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([1]))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))

        self.thr_fun = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x, z, PhiWeight, PhiTWeight, PhiTb):

        x = x - self.lambda_step * PhiTPhi_fun(x, PhiWeight, PhiTWeight)
        x = x + self.lambda_step * PhiTb
        x_input = x
        x_input = torch.cat([x_input, z], 1)

        x = F.conv2d(x_input, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        soft_thr = self.soft_thr * self.thr_fun(x_forward)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)

        x_pred = x_input + x_backward

        return x_pred

# Define ISTA-Net-plus
class SODASNet(torch.nn.Module):
    def __init__(self, LayerNo, n_input):
        super(SODASNet, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        n_feat = 32 - 1

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)
        self.fe = nn.Conv2d(1, n_feat, 3, padding=1, bias=True)

    def forward(self, Phix, Phi):
      
        PhiWeight = Phi.contiguous().view(n_input, 1, 33, 33)
        # Initialization-subnet
        PhiTWeight = Phi.t().contiguous().view(n_output, n_input, 1, 1)
        
        PhiTb = F.conv2d(Phix, PhiTWeight, padding=0, bias=None)
        PhiTb = torch.nn.PixelShuffle(33)(PhiTb)
        x = PhiTb    # Conduct initialization

        z = self.fe(x)

        for i in range(self.LayerNo):
            x_dual = self.fcs[i](x, z, PhiWeight, PhiTWeight, PhiTb)
            x = x_dual[:, :1, :, :]
            z = x_dual[:, 1:, :, :]

#         x_final = x.view(-1, 1089)
        x_final = x

        return x_final

def PhiTPhi_fun(x, PhiW, PhiTW):
    temp = F.conv2d(x, PhiW, padding=0,stride=33, bias=None)
    temp = F.conv2d(temp, PhiTW, padding=0, bias=None)
    return torch.nn.PixelShuffle(33)(temp)

model = SODASNet(layer_num, n_input)
model = nn.DataParallel(model)
model = model.to(device)

num_params = 0
for para in model.parameters():
    num_params += para.numel()
print("total para num: %d\n" %num_params)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.load_state_dict(torch.load('./%s/CS_%s_ratio_%d.pkl' % (args.model_dir, args.algo_name, cs_ratio)))

def rgb2ycbcr(rgb):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = rgb.shape
    if len(shape) == 3:
        rgb = rgb.reshape((shape[0] * shape[1], 3))
    ycbcr = np.dot(rgb, m.transpose() / 255.)
    ycbcr[:,0] += 16.
    ycbcr[:,1:] += 128.
    return ycbcr.reshape(shape)

# ITU-R BT.601
# https://en.wikipedia.org/wiki/YCbCr
# YUV -> RGB
def ycbcr2rgb(ycbcr):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    rgb = copy.deepcopy(ycbcr)
    rgb[:,0] -= 16.
    rgb[:,1:] -= 128.
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    return rgb.clip(0, 255).reshape(shape)

def imread_CS_py(Iorg):
    block_size = 33
    [row, col] = Iorg.shape
    row_pad = block_size-np.mod(row,block_size)
    col_pad = block_size-np.mod(col,block_size)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col+col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]


def img2col_py(Ipad, block_size):
    [row, col] = Ipad.shape
    row_block = row/block_size
    col_block = col/block_size
    block_num = int(row_block*col_block)
    img_col = np.zeros([block_size**2, block_num])
    count = 0
    for x in range(0, row-block_size+1, block_size):
        for y in range(0, col-block_size+1, block_size):
            img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].reshape([-1])
            # img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].transpose().reshape([-1])
            count = count + 1
    return img_col


def col2im_CS_py(X_col, row, col, row_new, col_new):
    block_size = 33
    X0_rec = np.zeros([row_new, col_new])
    count = 0
    for x in range(0, row_new-block_size+1, block_size):
        for y in range(0, col_new-block_size+1, block_size):
            X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size])
            # X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size]).transpose()
            count = count + 1
    X_rec = X0_rec[:row, :col]
    return X_rec


def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


test_dir = os.path.join(args.data_dir, test_name)
if test_name=='Set11':
    filepaths = glob.glob(test_dir + '/*.tif')
if test_name=='bsd68' or test_name=='CBSD68':
    filepaths = glob.glob(test_dir + '/*.png')
if test_name=='Urban100':
    filepaths = glob.glob(test_dir + '/*.png')
    

result_dir = os.path.join(args.result_dir, test_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


ImgNum = len(filepaths)
PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)


Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor)
Phi = Phi.to(device)

results_csv=[]

print('\n')
print("CS Reconstruction Start")

with torch.no_grad():
    for img_no in range(ImgNum):

        imgName = filepaths[img_no]
        if test_name=='Set11': 
            img_index = imgName.split('/')[-1].split('.')[0]
        if test_name=='bsd68' or test_name=='CBSD68': 
            img_index = imgName.split('_')[-1].split('.')[0][-2:]
        if test_name=='Urban100':
            img_index = imgName.split('_')[-1].split('.')[0][-3:]
        Img = cv2.imread(imgName, 1)

        Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
        Img_rec_yuv = Img_yuv.copy()

        Iorg_y = Img_yuv[:,:,0]
        Iorg = Iorg_y.copy()

        [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(Iorg_y)
#         Icol = img2col_py(Ipad, 33).transpose()/255.0

        Img_output = Ipad.reshape(1, 1, Ipad.shape[0], Ipad.shape[1])
        torch.cuda.synchronize()
        start = time()

        batch_x = torch.from_numpy(Img_output)
        batch_x = batch_x.type(torch.FloatTensor)
        batch_x = batch_x.to(device)
        
        PhiWeight = Phi.contiguous().view(n_input, 1, 33, 33)
        Phix = F.conv2d(batch_x, PhiWeight, padding=0, stride=33, bias=None)
        print("Phix:", Phix.shape, torch.min(Phix), torch.max(Phix))
        noise_sigma = noise * torch.randn_like(Phix)
        Phix = Phix+noise_sigma

        x_output = model(Phix/255.0, Phi)
        torch.cuda.synchronize()
        end = time()

#         x_output = x_output
        Prediction_value = x_output.cpu().data.numpy().squeeze()
        row = Iorg.shape[0]
        col = Iorg.shape[1]

        X_rec = np.clip(Prediction_value[0:row, 0:col], 0, 1)

        rec_PSNR = psnr(X_rec*255, Iorg.astype(np.float64))
        rec_SSIM = ssim(X_rec*255, Iorg.astype(np.float64), data_range=255)

        print("[%02d/%02d] Run time for %s is %.4f, PSNR is %.2f, SSIM is %.4f" % (img_no, ImgNum, imgName, (end - start), rec_PSNR, rec_SSIM))

        Img_rec_yuv[:,:,0] = X_rec*255

        im_rec_rgb = cv2.cvtColor(Img_rec_yuv, cv2.COLOR_YCrCb2BGR)
        im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)

        cv2.imwrite("%s/%s_%s_layer_%d_ratio_%d_PSNR_%.2f_SSIM_%.4f.png" % (
        result_dir, img_index, args.algo_name, layer_num, cs_ratio, rec_PSNR, rec_SSIM), im_rec_rgb)

        del x_output
        
        result_csv = [img_index] + [rec_PSNR] + [rec_SSIM]
        results_csv.append(result_csv)

        PSNR_All[0, img_no] = rec_PSNR
        SSIM_All[0, img_no] = rec_SSIM

print('\n')
output_data = "CS ratio is %d, Avg PSNR/SSIM for %s is %.2f/%.4f \n" % (cs_ratio, args.test_name, np.mean(PSNR_All), np.mean(SSIM_All))
print(output_data)

print("CS Reconstruction End")