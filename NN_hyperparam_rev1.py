'''
Hyperparameter adjustment for the CNN-LSTM
'''
from hyperopt import fmin, tpe, hp, rand
import argparse
import sys
import os
import random
import numpy as np
import copy
import scipy.io

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import utils
from utils import GLO
import datasets
import Net
import Net_rev1
import U_Net
import UNet

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
cudnn.deterministic = True

class OPT():
    cuda = 0
    seq_len = 4
    checkpoint_every = 50
    num = 1
    interval = 3
    if num==1:        
        netpath = '_opt_param1'
    elif num==3:        
        if interval == 3:
            netpath = '_3s_opt_param1'
        if interval == 5:
            netpath = '_5s_opt_param1'

parameter_space = {
    'epochs': hp.choice('epochs', [800,900,1000,1100,1200]),
    'lr': hp.choice('lr', [1e-4,2e-4,1e-3,2e-3]),
    'lambda': hp.choice('lambda', [0,1e-4,2e-4,1e-3,2e-3,1e-2,2e-2]),
    'batch_size': hp.choice('batch_size', [2,4,8,16,32,64,128]),
    'betas0': hp.choice('betas0', [0.5,0.6,0.7,0.8,0.9]),
    'dropout': hp.choice('dropout', [0,0.1,0.2,0.3,0.4,0.5]),
    'slope': hp.choice('slope', [0.01,0.05,0.1,0.15,0.2]),
    'fc_dim': hp.choice('fc_dim', [128,256,512,1024]),
    'hidden_dim': hp.choice('hidden_dim', [32,64,128,256,512]),
    'nf': hp.choice('nf', [4,8,16])
}

opt = OPT()
torch.cuda.set_device(opt.cuda)
GLO.set_value('seq_len', opt.seq_len)
num = opt.num
GLO.set_value('num', num)
utils.prepare_data(num, opt.netpath)

if num == 1:
    dataset_path = './input/cell_temp_'+str(num)+'C.mat'
else:
    dataset_path = './input/cell_temp_'+str(num)+'_'+str(opt.interval)+'s.mat'
data = scipy.io.loadmat(dataset_path)
data = data['cell_temperature']-273.15
data = data[:, :, 1:]  # to avoid initial value -> scale -> NaN/invalid
test_target = [data[:, :, i] for i in range(data.shape[-1])]

input = scipy.io.loadmat('./output_'+str(num) + opt.netpath +'/adjusted_input_'+str(num)+'.mat')
input = input['adjusted_input']
target = scipy.io.loadmat('./output_'+str(num) + opt.netpath +'/adjusted_target_'+str(num)+'.mat')
target = target['adjusted_target']

# consecutive input, form the seq_len input
input = utils.form_time_seq(input, opt.seq_len)
target = utils.form_time_seq(target, opt.seq_len)
test_target = utils.form_time_seq(test_target, opt.seq_len)
input, target = np.array(input), np.array(target)
test_target = np.array(test_target)

# data to tensor to pairs in loader
total_index = set(np.arange(0, input.shape[0]))
test_index_list = random.sample(range(0, input.shape[0]), input.shape[0]//10)

test_index = set(test_index_list)  # sequence changed
# scipy.io.savemat('./output_'+str(num) + opt.netpath +'/test_index_'+str(num)+'.mat',
                #  {'test_index': list(test_index)})
train_index = total_index-test_index

GLO.set_value('train_index', train_index)
GLO.set_value('test_index', test_index)

test_target_actual = test_target[list(test_index)]  # actual temp

def func(args):
    print(args)
    opt.epochs = args['epochs']
    opt.lr = args['lr']
    opt.weight_decay = args['lambda']
    opt.batch_size = args['batch_size']
    opt.betas0 = args['betas0']
    opt.dropout = args['dropout']
    opt.slope = args['slope']
    opt.fc_dim = args['fc_dim']
    opt.hidden_dim = args['hidden_dim']
    opt.nf = args['nf']

    GLO.set_value('batch_size', opt.batch_size)
    GLO.set_value('dropout', opt.dropout)
    GLO.set_value('slope', opt.slope)
    
    MyDataset = datasets.process_data(input, target)
    train_dataloader, test_dataloader = MyDataset.encap(num)

    test_iter = iter(test_dataloader)
    test_data = test_iter.next()
    test_input_cpu, test_target_cpu = test_data
    
    test_input = test_input_cpu.cuda()
    test_target = test_target_cpu.cuda()

    # --------------------------------------- initialize network ---------------------------------------
    inputChannelSize = 1*opt.seq_len
    outputChannelSize = 1*opt.seq_len
    # net = Net.net(inputChannelSize, outputChannelSize, opt.nf)
    # net = Net_rev1.net(inputChannelSize, outputChannelSize, opt.nf, opt.hidden_dim, opt.fc_dim)
    # net = U_Net.net(inputChannelSize, outputChannelSize, opt.nf)
    net = UNet.net(inputChannelSize, outputChannelSize, opt.nf, opt.hidden_dim, opt.fc_dim)
    # print(net)

    net.apply(utils.weights_init)
    criterion = nn.MSELoss()

    net.cuda()
    criterion.cuda()

    # --------------------------------------- training process ---------------------------------------
    if os.path.isdir('./network_'+str(opt.num) + opt.netpath) == False:
        os.makedirs('./network_'+str(opt.num) + opt.netpath)
    if os.path.isdir('./output_'+str(num) + opt.netpath) == False:
        os.makedirs('./output_'+str(num) + opt.netpath)
    if os.path.isdir('./test_output_process_'+str(num) + opt.netpath) == False:
        os.makedirs('./test_output_process_'+str(num) + opt.netpath)
    
    optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(
        opt.betas0, 0.999), weight_decay=opt.weight_decay)

    net.train()
    test_output_process = {}
    for epoch in range(opt.epochs):
        for i, data in enumerate(train_dataloader, 0):
            train_input_cpu, train_target_cpu = data  # data[0] data[1]
            train_input, train_target = train_input_cpu.cuda(), train_target_cpu.cuda()

            output, _, _ = net(train_input.float())
            loss = criterion(output.float(), train_target.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_iter_num = epoch * len(train_dataloader) + i + 1
            if (global_iter_num == opt.epochs * len(train_dataloader)):
                # print('[%d/%d][%d/%d] %d loss: %f'
                #       % (epoch, opt.epochs, i, len(train_dataloader), global_iter_num, loss.item()))                
                
                with torch.no_grad():
                    net.eval()
                    test_output, _, _ = net(test_input.float())
                    test_loss = criterion(test_output.float(), test_target.float())
                net.train()

                if (global_iter_num == opt.epochs * len(train_dataloader)):
                    test_output = test_output[:, 0].cpu().numpy()
                    test_output = utils.square2rec(test_output)  # contain denormalization
                    test_output_process['test_output_' +
                                        str(global_iter_num)] = test_output                

        # do checkpointing
        if (epoch % opt.checkpoint_every == 0) or (epoch == opt.epochs - 1):
            torch.save(net.state_dict(), '%s/net_epoch_%d.pth' %
                    ('network_'+str(num)+opt.netpath, epoch))

        # store the last test result
        if epoch == opt.epochs-1:
            with torch.no_grad():
                net.eval()
                test_output, _, _ = net(test_input.float())
            test_output = test_output.cpu().numpy()
            test_output = utils.square2rec(test_output)  # contain denormalization
            # scipy.io.savemat('./test_output_process_'+str(num)+ opt.netpath +'/test_output.mat',
            #                 {'test_output': test_output})

    # scipy.io.savemat('./test_output_process_'+str(num)+ opt.netpath +'/test_output_process_' +
    #                 str(num)+'.mat', {'test_output_process': test_output_process})

    # shape (100,50,34) (33,4,100,50)
    test_error = abs(test_output-test_target_actual)
    ele_num = np.prod([test_error.shape[-2], test_error.shape[-1]])
    # print('ele_num: ', ele_num)
    test_error_avg = [np.sum(test_error[i])/opt.seq_len /
                    ele_num for i in range(test_error.shape[0])]
    test_error_max = [np.max(test_error[i]) for i in range(test_error.shape[0])]
    # print('test_error_avg: ', test_error_avg, '\n', 'max(test_error_avg): ', max(
    #     test_error_avg), '\n', 'max(test_error_max): ', max(test_error_max))

    return max(test_error_avg)*1000+max(test_error_max)*1000


best = fmin(func, parameter_space, algo=tpe.suggest, max_evals=100)
print(best)

with open('./NN_'+str(opt.num)+opt.netpath+'.txt','a') as f:
    for idx,val in best.items():
        f.write(str(val)+' ')
        val=round(val,4)
        best[idx]=val
    f.write('\n')
    f.write(str(best)+'\n')