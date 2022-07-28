import argparse
import os
import numpy as np
import scipy.io
import random

import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

import utils
from utils import GLO
import Net
import Net_rev1
import UNet
import pid

import time

SEED = 27
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
cudnn.deterministic = True

debug = 0


class OPT():
    cuda = 0
    batch_size = 1
    seq_len = 4
    num = 3
    interval = 5
    if num==1:
        network = 799
        test_pid = 300
        delta_t = 30
        steps = 60
        netpath = '_opt_param1'
    elif num==3:    
        if interval == 3:
            network = 899
            test_pid = 300
            delta_t = 30
            steps = 60
            netpath = '_3s_opt_param1'
        if interval == 5:
            network = 899
            test_pid = 170
            delta_t = 20
            steps = 20
            netpath = '_5s_opt_param1'

    with open('./NN_'+str(num)+netpath+'.txt','r') as f:
        line = f.readlines()
    param = eval(line[-1])
    dropout=param['dropout']
    fc_dim=param['fc_dim']
    hidden_dim=param['hidden_dim']
    nf=param['nf']
    slope=param['slope']

if not debug:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=4)    
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--interval', type=int, default=3)
    parser.add_argument('--netpath', type=str, default='_param1')
    parser.add_argument('--network', type=int, default=799)
    parser.add_argument('--test_pid', type=int, default=300)
    parser.add_argument('--delta_t', type=int, default=30)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--w_ID', type=int, nargs='+')

    parser.add_argument('--nf', type=int, default=16)
    parser.add_argument('--betas0', type=float, default=0.9)
    parser.add_argument('--betas1', type=float, default=0.999)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--slope', type=float, default=0.2)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--fc_dim', type=int, default=512)

    opt = parser.parse_args()
else:
    opt = OPT()      

with open('./NN_'+str(opt.num)+opt.netpath+'.txt','r') as f:
        line = f.readlines()
param = eval(line[-1])
opt.dropout=param['dropout']
opt.fc_dim=param['fc_dim']
opt.hidden_dim=param['hidden_dim']
opt.nf=param['nf']
opt.slope=param['slope']

torch.cuda.set_device(opt.cuda)
GLO.set_value('batch_size', opt.batch_size)
GLO.set_value('seq_len', opt.seq_len)
num = opt.num
GLO.set_value('num', num)
GLO.set_value('dropout', opt.dropout)
GLO.set_value('hidden_dim', opt.hidden_dim)
GLO.set_value('slope', opt.slope)

utils.prepare_data(num, opt.netpath)

# --------------------------------------- normalized data ---------------------------------------
input = scipy.io.loadmat('./output_'+str(num) + opt.netpath +'/adjusted_input_'+str(num)+'.mat')
target = scipy.io.loadmat('./output_'+str(num) + opt.netpath +'/adjusted_target_'+str(num)+'.mat')
input = input['adjusted_input']  # normalized
target = target['adjusted_target']

test_pid = opt.test_pid

# --------------------------------------- load pre-trained network ---------------------------------------
inputChannelSize = 1*opt.seq_len
outputChannelSize = 1*opt.seq_len
# net = Net.net(inputChannelSize, outputChannelSize, nf)
# net = Net_rev1.net(inputChannelSize, outputChannelSize, opt.nf, opt.hidden_dim, opt.fc_dim)
# net = U_Net.net(inputChannelSize, outputChannelSize, opt.nf)
net = UNet.net(inputChannelSize, outputChannelSize, opt.nf, opt.hidden_dim, opt.fc_dim)
net.load_state_dict(torch.load('./network_'+str(opt.num) + opt.netpath +
                    '/net_epoch_'+str(opt.network)+'.pth'))
net.cuda()
net.eval()

# if os.path.isfile('./output/network_sensitivity.txt')==False:
#     utils.test_network_sensitivity(input,test_pid,target,net)

steps = opt.steps  # chosen & no need to be the same as delta_t
pid_algorithm = pid.pid_algorithm(opt, steps, input, target, net)
error_wo_avg = pid_algorithm.without_pid(test_pid)
scipy.io.savemat('./output_'+str(num)+opt.netpath+'/error_wo_avg.mat',{'error_wo_avg':error_wo_avg})
