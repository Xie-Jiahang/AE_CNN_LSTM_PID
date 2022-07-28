'''
TPE algorithm to adjust PID parameters in offline mode
in paper: take average of optimal PID parameters based on delta_t samples w/ target (for each sample do TPE)
in code: the optimal PID parameters based on prediction w/o target at 'current' time step t (a 'fake' online time step)
'''
from hyperopt import fmin, tpe, hp, rand
import numpy as np
import scipy.io
import random

import torch
import torch.backends.cudnn as cudnn

import utils
from utils import GLO
import Net
import Net_rev1
import U_Net
import UNet
import pid
import error_fitting
import linear_regression

SEED = 27
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
cudnn.deterministic = True


class OPT():
    cuda = 0
    batch_size = 1
    seq_len = 4  
    nf = 16
    hidden_dim = 32
    fc_dim = 128
    dropout = 0.2
    slope = 0.1
    Kp = 0
    Ki = 0
    Kd = 0
    magnitude = 0
    num = 3
    interval = 5
    if num==1:
        network = 1099 # 799        
        test_pid = 300
        delta_t = 30
        steps = 60
        netpath = '_opt_param1'
    elif num==3:
        # network = 1199
        if interval == 3:
            network = 999 # 1199
            test_pid = 300
            delta_t = 30
            steps = 60
            netpath = '_3s_opt_param1'
        if interval == 5:
            network = 899 # 1199
            test_pid = 170
            delta_t = 20
            steps = 20
            netpath = '_5s_opt_param1'

parameter_space_PID = {
    'Kp': hp.uniform('Kp', 1e-3, 2),
    'Ki': hp.uniform('Ki', 0, 2),
    'Kd': hp.uniform('Kd', 0, 2),
    'magnitude': hp.uniform('magnitude', 1e-3, 1)
}

opt = OPT()
torch.cuda.set_device(opt.cuda)
GLO.set_value('batch_size', opt.batch_size)
GLO.set_value('seq_len', opt.seq_len)

num = opt.num
GLO.set_value('num', num)
GLO.set_value('dropout', opt.dropout)
GLO.set_value('hidden_dim', opt.hidden_dim)
GLO.set_value('slope', opt.slope)

utils.prepare_data(num, opt.netpath)

input = scipy.io.loadmat('./output_'+str(num) + opt.netpath +'/adjusted_input_'+str(num)+'.mat')
target = scipy.io.loadmat('./output_'+str(num) + opt.netpath +'/adjusted_target_'+str(num)+'.mat')
input = input['adjusted_input']  # normalized, for each time step, not organized as for each seq
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

steps = opt.steps  # delta_t: sample number; steps: PID response step
pid_algorithm = pid.pid_algorithm(opt, steps, input, target, net)


def func(args):
    print(args)
    opt.Kp = args['Kp']
    opt.Ki = args['Ki']
    opt.Kd = args['Kd']
    opt.magnitude = args['magnitude']
    
    # generate data for fitting error curve
    # t: current time step
    t = opt.test_pid
    # smaller delta_t leads to higher accuracy in predict error, becoz closer to the time step now
    delta_t = opt.delta_t

    error_with_target = []
    error_t_avg = []    
    for i in range(t-2*delta_t+1, t-delta_t+1):
        error_with_target.append(pid_algorithm.pid_with_target(i)[0])
        error_t_avg.append(pid_algorithm.pid_with_target(i)[1])

    scipy.io.savemat('./output_'+str(num)+opt.netpath+'/error_with_target.mat',
                 {'error_with_target': error_with_target})    

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

    predict_result_seq = []
    for k in range(steps):
        tmp = np.mean(predict_result[k, -1, -1, :, :])
        predict_result_seq.append(tmp)
    target_osc = 0 # oscillation of prediction response compared to target response
    predict_osc = 0
    for k in range(steps-1):
        target_osc += abs(target_seq[k+1]-target_seq[k])
        predict_osc += abs(predict_result_seq[k+1]-predict_result_seq[k])    
    # 1000: estimated based on magnitude level
    return final_prediction_error+1000*predict_osc/target_osc


best = fmin(func, parameter_space_PID, algo=tpe.suggest, max_evals=100)
print(best)

with open('./hyperopt_'+str(opt.num)+opt.netpath+'.txt','a') as f:
    for idx,val in best.items():
        f.write(str(val)+' ')
        val=round(val,4)
        best[idx]=val
    f.write('\n')
    f.write(str(best)+'\n')