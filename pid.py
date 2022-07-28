import numpy as np
import copy
import scipy.io

import torch
import matplotlib.pyplot as plt
import pynvml

import utils
from utils import GLO

import time

# --------------------------------------- PID algorithm ---------------------------------------


class pid_algorithm():
    def __init__(self, opt, steps, input, target, net):
        self.opt = opt
        self.steps = steps
        self.input = input # for each time step, not organized as for each seq
        self.target = target
        self.net = net

    def without_pid(self, test_pid): # direct propagation of LSTM internal states along time, w/o PID
        batch_size = GLO.get_value('batch_size')
        seq_len = GLO.get_value('seq_len')
        num = GLO.get_value('num')
        hidden_dim = GLO.get_value('hidden_dim')
        scale = self.target.shape[1]

        y_t = torch.from_numpy(self.target[test_pid-seq_len+1:test_pid+1])
        y_t = y_t.unsqueeze(0)
        yout = torch.zeros((self.steps, batch_size, seq_len, scale, scale))

        # without internal states evolution, direct prediction at t, with only the last sequence as input [X(t-seq_len),...,X(t)]
        test_input = torch.from_numpy(
            self.input[test_pid-seq_len+1:test_pid+1])
        test_input = test_input.unsqueeze(0)
        direct_prediction, _, _ = self.net(test_input.cuda().float())
        direct_prediction = direct_prediction.cpu().detach().numpy()
        direct_prediction = utils.square2rec(direct_prediction)
        scipy.io.savemat('./output_'+str(num) + self.opt.netpath +
                         '/direct_prediction.mat', {'yout': direct_prediction})
        tmp_target = y_t.numpy()
        tmp_target = utils.square2rec(tmp_target)
                
        direct_prediction = direct_prediction[:,-1,:,:]        
        tmp_target = tmp_target[:,-1,:,:]
        direct_prediction_error = np.sum(abs(direct_prediction-tmp_target))/np.prod(tmp_target.shape)
        print('direct_prediction_error: %.4f' %direct_prediction_error)        

        # prediction with iteration of states
        # target has the real value for each seq, not only the final seq
        target = torch.zeros((self.steps, batch_size, seq_len, scale, scale))
        starttime=time.time()
        for k in range(self.steps):
            index = test_pid-seq_len*(self.steps-k)+1
            s = torch.from_numpy(self.input[index:index+seq_len])
            s = s.unsqueeze(0)
            target[k] = torch.from_numpy(self.target[index:index+seq_len])

            if k == 0:
                h, c = torch.zeros(1, batch_size, hidden_dim).cuda(
                ), torch.zeros(1, batch_size, hidden_dim).cuda()

            tmp, h, c = self.net(s.cuda().float(), h, c)
            yout[k] = tmp.cpu().detach()

        prediction = utils.square2rec(yout[-1].numpy())
        tmp_target = target[-1].numpy()
        tmp_target = utils.square2rec(tmp_target)        
                
        prediction = prediction[:,-1,:,:]
        tmp_target = tmp_target[:,-1,:,:]        
        woPID_error = np.sum(abs(prediction-tmp_target))/np.prod(tmp_target.shape)
        print('woPID_error: %.4f' %woPID_error)

        yout = yout.numpy()
        
        result = utils.square2rec(yout)
        scipy.io.savemat('./output_'+str(num) + self.opt.netpath +
                         '/woPID_yout.mat', {'yout': result})
        endtime=time.time()
        print('Prediction time: %.4f' %(10*(endtime-starttime)))
        
        target = target.numpy()        
        real_error_normalized = np.zeros(
            (self.steps, batch_size, seq_len, scale, scale))
        for k in range(self.steps):            
            real_error_normalized[k] = target[k]-yout[k]
        target = utils.square2rec(target)

        # list of error, in each internal states transfer iteration w/o PID
        error_wo_avg = [np.mean(abs(result[i, :, -1]-target[i, :, -1]))
                        for i in range(self.steps)]

        return error_wo_avg

    def pid_with_target(self, test_pid): # LSTM internal states transfer along time, w/ PID & target (offline)
        batch_size = GLO.get_value('batch_size')
        seq_len = GLO.get_value('seq_len')
        num = GLO.get_value('num')
        hidden_dim = GLO.get_value('hidden_dim')
        scale = self.target.shape[1]

        # the normalized error without magnitude limit
        real_error_normalized = torch.zeros(
            (self.steps, batch_size, seq_len, scale, scale))
        x = torch.zeros((3, batch_size, seq_len, scale, scale))
        real_error_normalized_1 = torch.zeros(
            (batch_size, seq_len, scale, scale))
        real_error_normalized_2 = torch.zeros(
            (batch_size, seq_len, scale, scale))

        u_1 = torch.zeros((batch_size, seq_len, scale, scale))
        du = torch.zeros((batch_size, seq_len, scale, scale))
        u = torch.zeros((self.steps, batch_size, seq_len, scale, scale))

        yout = torch.zeros((self.steps, batch_size, seq_len, scale, scale))

        # pynvml.nvmlInit()
        # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

        # target has the real value for each seq, not only the final seq
        target = torch.zeros((self.steps, batch_size, seq_len, scale, scale))
        for k in range(self.steps):            
            # print(meminfo.used)

            index = test_pid-seq_len*(self.steps-k)+1
            s = torch.from_numpy(self.input[index:index+seq_len])
            s = s.unsqueeze(0)
            y_t = torch.from_numpy(self.target[index:index+seq_len])
            y_t = y_t.unsqueeze(0)
            target[k] = torch.from_numpy(self.target[index:index+seq_len])

            test_input = s+u_1 # equal to s[0,0]+u_1
            if k == 0:
                h, c = torch.zeros(1, batch_size, hidden_dim).cuda(
                ), torch.zeros(1, batch_size, hidden_dim).cuda()

            tmp, h, c = self.net(test_input.cuda().float(), h, c)
            yout[k] = tmp.cpu().detach()

            real_error_normalized[k] = y_t-yout[k]
            x[0] = real_error_normalized[k]-real_error_normalized_1
            x[1] = real_error_normalized[k]-2 * \
                real_error_normalized_1+real_error_normalized_2
            x[2] = real_error_normalized[k]
            
            du = self.opt.Kp*x[0]+self.opt.Kd*x[1]+self.opt.Ki*x[2]
            u[k] = u_1+du # the other method calculate u based on real_error_normalized is not right, since no consider for previous time step limit on u
            # u[k]=self.opt.Kp*real_error_normalized[k]+self.opt.Ki*torch.sum(real_error_normalized[:k+1],dim=0)+self.opt.Kd*(real_error_normalized[k]-real_error_normalized_1)
            u[k] = torch.where(u[k] <= self.opt.magnitude,
                               u[k], torch.tensor([self.opt.magnitude]).float())
            u[k] = torch.where(u[k] >= -self.opt.magnitude,
                               u[k], -torch.tensor([self.opt.magnitude]).float())

            u_1 = u[k]
            real_error_normalized_1 = real_error_normalized[k]

        yout = yout.numpy()

        real_error_normalized = real_error_normalized.numpy()
        # the normalized error with magnitude limit
        error_normalized = copy.deepcopy(real_error_normalized)
        # the denormalized error with magnitude limit
        error = utils.error_square2rec(error_normalized, 1)        

        # calculate real_error_normalized: abs(result-target), becoz matrix real_error_normalized has been limited the magnitude (now not, limit u magnitude)
        result = utils.square2rec(yout)
        
        target = target.numpy()
        for k in range(self.steps):
            # the normalized error without magnitude limit
            real_error_normalized[k] = target[k]-yout[k]
        target = utils.square2rec(target)

        error_t_avg = [np.sum(abs(result[i, :, -1]-target[i, :, -1])) /
                       np.prod(target.shape[-2:]) for i in range(self.steps)]  # the denormalized error without magnitude limit
        
        # list of error, in each internal states transfer iteration w/ PID & known target
        error_t_avg = [np.mean(abs(result[i, :, -1]-target[i, :, -1]))
                       for i in range(self.steps)]

        return real_error_normalized, error_t_avg, result

    def pid_with_error(self, t, predict_error_normalized): # LSTM internal states transfer along time, w/ PID, w/o target for any time step (online)
        batch_size = GLO.get_value('batch_size')
        seq_len = GLO.get_value('seq_len')
        num = GLO.get_value('num')
        hidden_dim = GLO.get_value('hidden_dim')
        scale = self.target.shape[1]

        predict_error_normalized = torch.from_numpy(predict_error_normalized)
        x = torch.zeros((3, batch_size, seq_len, scale, scale))
        predict_error_normalized_1 = torch.zeros(
            (batch_size, seq_len, scale, scale))
        predict_error_normalized_2 = torch.zeros(
            (batch_size, seq_len, scale, scale))

        u_1 = torch.zeros((batch_size, seq_len, scale, scale))
        du = torch.zeros((batch_size, seq_len, scale, scale))
        u = torch.zeros((self.steps, batch_size, seq_len, scale, scale))

        yout = torch.zeros((self.steps, batch_size, seq_len, scale, scale))

        # target has the real value for each seq, not only the final seq
        target = torch.zeros((self.steps, batch_size, seq_len, scale, scale))        
        
        # starttime=time.time()
        duration=0

        for k in range(self.steps):
            index = t-seq_len*(self.steps-k)+1
            s = torch.from_numpy(self.input[index:index+seq_len])
            s = s.unsqueeze(0)
            y_t = torch.from_numpy(self.target[index:index+seq_len])
            y_t = y_t.unsqueeze(0)
            target[k] = torch.from_numpy(self.target[index:index+seq_len])
            
            test_input = s+u_1  # equal to s[0,0]+u_1
            if k == 0:
                h, c = torch.zeros(1, batch_size, hidden_dim).cuda(
                ), torch.zeros(1, batch_size, hidden_dim).cuda()

            starttime=time.time()
            tmp, h, c = self.net(test_input.cuda().float(), h, c)
            endtime=time.time()
            duration += (endtime-starttime)
            
            yout[k] = tmp.cpu().detach()

            x[0] = predict_error_normalized[k]-predict_error_normalized_1
            x[1] = predict_error_normalized[k]-2 * \
                predict_error_normalized_1+predict_error_normalized_2
            x[2] = predict_error_normalized[k]
            
            # if k<self.steps-1:
            #     predict_error_normalized[k] = y_t-yout[k]

            du = self.opt.Kp*x[0]+self.opt.Kd*x[1]+self.opt.Ki*x[2]
            u[k] = u_1+du            
            u[k] = torch.where(u[k] <= self.opt.magnitude,
                               u[k], torch.tensor([self.opt.magnitude]).float())
            u[k] = torch.where(u[k] >= -self.opt.magnitude,
                               u[k], -torch.tensor([self.opt.magnitude]).float())

            u_1 = u[k]
            predict_error_normalized_1 = predict_error_normalized[k]

        # endtime=time.time()
        print('Prediction time: %.4f'%(10*duration)) # roughly estimated as 10 2D layers to solve 3D temperature distribution

        final_prediction = utils.square2rec(yout[-1].numpy())
        tmp_target = target[-1].numpy()
        tmp_target = utils.square2rec(tmp_target)        
                
        final_prediction = final_prediction[:,-1,:,:]
        tmp_target = tmp_target[:,-1,:,:]        
        final_prediction_error = np.sum(abs(final_prediction-tmp_target))
        print('final_prediction_error: %.4f' %(final_prediction_error/np.prod(tmp_target.shape)))

        yout = yout.numpy()

        predict_error_normalized = predict_error_normalized.numpy()
        error_normalized = copy.deepcopy(predict_error_normalized)
        error = utils.error_square2rec(error_normalized, 1)

        # the real predict_error_normalized: abs(result-target), becoz matrix predict_error_normalized has been limited the magnitude (now not, limit u magnitude)
        result = utils.square2rec(yout)        
        scipy.io.savemat('./output_'+str(num)+ self.opt.netpath +'/yout.mat', {'yout': result})

        target = target.numpy()
        for k in range(self.steps):
            # the normalized error without magnitude limit
            predict_error_normalized[k] = target[k]-yout[k]
        target = utils.square2rec(target)
        target_seq = []
        for k in range(self.steps):
            tmp = np.mean(target[k, -1, -1, :, :])
            target_seq.append(tmp)

        # list of error, in each PID iteration w/ predicted error, w/o target
        error_p_avg = [np.mean(abs(result[i, :, -1]-target[i, :, -1]))
                       for i in range(self.steps)]

        return result, error_p_avg, final_prediction_error, target_seq
