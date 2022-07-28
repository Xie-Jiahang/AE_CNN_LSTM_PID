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
import U_Net
import UNet
import pid
import error_fitting
import linear_regression

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
    nf = 16
    hidden_dim = 128
    fc_dim = 512  
    dropout = 0.5
    slope = 0.2
    num = 3
    interval = 3
    if num==1:
        network = 799
        test_pid = 300
        delta_t = 30
        steps = 60
        netpath = '_Urev1_param1'
    elif num==3:
        network = 1199
        if interval == 3:
            test_pid = 300
            delta_t = 30
            steps = 60
            netpath = '_3s_Urev1_param2'
        if interval == 5:
            test_pid = 170
            delta_t = 20
            steps = 20
            netpath = '_5s_Urev1_param1'

    with open('./hyperopt_'+str(num)+netpath+'.txt','r') as f:
        line = f.readlines()    
    param = list(map(float,line[0].split()))
    Kp = param[2]
    Ki = param[1]
    Kd = param[0]
    magnitude = param[-1]

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
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--slope', type=float, default=0.2)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--fc_dim', type=int, default=512)

    opt = parser.parse_args()
else:
    opt = OPT()      

with open('./hyperopt_'+str(opt.num)+opt.netpath+'.txt','r') as f:
    line = f.readlines()
param = list(map(float,line[0].split()))
opt.Kp = param[2]
opt.Ki = param[1]
opt.Kd = param[0]
opt.magnitude = param[-1]

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
# net = Net.net(inputChannelSize, outputChannelSize, opt.nf)
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

# get result of offline temperature propogation with PID, w_ID: the last/current time step
if opt.w_ID:
    result = {}
    for i in opt.w_ID:
        result['result_' + str(i)] = pid_algorithm.pid_with_target(test_pid)[2]
    scipy.io.savemat('./output_'+str(num)+ opt.netpath +'/wPID_yout.mat', {'yout': result})

# --------------------------------------- store 0~t-1 error matrices ---------------------------------------
# generate data for fitting error curve
t = opt.test_pid # the last/current time step is t
# smaller delta_t leads to higher accuracy in predict error, becoz closer to the time step now
delta_t = opt.delta_t

error_with_target = []
error_t_avg = []
for i in range(t-2*delta_t+1, t-delta_t+1): # collect error historical data between [t-2\Delt t, t-\Delta t]
    error_with_target.append(pid_algorithm.pid_with_target(i)[0])
    error_t_avg.append(pid_algorithm.pid_with_target(i)[1])

scipy.io.savemat('./output_'+str(num)+opt.netpath+'/error_with_target.mat',
                 {'error_with_target': error_with_target})
scipy.io.savemat('./output_'+str(num)+opt.netpath+'/error_t_avg.mat',{'error_t_avg':error_t_avg})

# error_predict=error_fitting.error_predict(delta_t, steps, opt.netpath)
# all_reg_model=error_predict.fitting()
# error_predict_result=error_predict.predict(all_reg_model)

# scipy.io.savemat('./output_'+str(num)+opt.netpath+'/error_predict_result.mat',{'error_predict_result':error_predict_result})

error_predict = linear_regression.error_predict(delta_t, steps, opt.netpath)
A, C = error_predict.fitting()
error_predict_result = error_predict.predict(A, C)

scipy.io.savemat('./output_'+str(num)+opt.netpath+'/error_predict_result.mat',
                 {'error_predict_result': error_predict_result})

error_predict_result = scipy.io.loadmat(
    './output_'+str(num)+opt.netpath+'/error_predict_result.mat')
error_predict_result = error_predict_result['error_predict_result']

predict_result, error_p_avg, final_prediction_error, target_seq = pid_algorithm.pid_with_error(
    t, error_predict_result)
scipy.io.savemat('./output_'+str(num)+opt.netpath+'/error_p_avg.mat',{'error_p_avg':error_p_avg})
