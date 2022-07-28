'''
train CNN-LSTM (base version)
'''
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

from tensorboardX import SummaryWriter
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

debug = 0


class OPT():
    batch_size = 4
    seq_len = 4
    cuda = 1
    epochs = 10
    checkpoint_every = 50
    num = 1


if not debug:
    parser = argparse.ArgumentParser()    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--seq_len', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--checkpoint_every', default=50,
                        help='number of epochs after which saving checkpoints')
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--interval', type=int, default=3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--netpath', type=str, default='_param1')
    
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

torch.cuda.set_device(opt.cuda)

# --------------------------------------- prepare data ---------------------------------------
# define global variables
GLO.set_value('batch_size', opt.batch_size)
GLO.set_value('seq_len', opt.seq_len)
num = opt.num
GLO.set_value('num', num)
GLO.set_value('dropout', opt.dropout)
GLO.set_value('slope', opt.slope)

# creat sensor data if needed
if num == 1:
    dataset_path = './input/cell_temp_'+str(num)+'C.mat'
else:
    dataset_path = './input/cell_temp_'+str(num)+'_'+str(opt.interval)+'s.mat'
data = scipy.io.loadmat(dataset_path)
data = data['cell_temperature']-273.15
data = data[:, :, 1:]  # to avoid initial value -> scale -> NaN/invalid

utils.prepare_sensor(data, num, opt.netpath)

# load created sensor data
sensor = scipy.io.loadmat('./output_'+str(num) + opt.netpath +'/sensor_2D_'+str(num)+'.mat')
sensor = sensor['sensor']

mean, std = data.mean(), data.std()
GLO.set_value('mean', mean)
GLO.set_value('std', std)
# print('mean: %.4f' %mean, '\n', 'std: %.4f' %std)

# get raw input & output
input = [sensor[:, :, i] for i in range(sensor.shape[-1])]
input = np.array(input)
# print(len(input))
target = [data[:, :, i] for i in range(data.shape[-1])]
target = np.array(target)
test_target = copy.deepcopy(target)

utils.save_adjusted(input, target, num, opt.netpath)
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
scipy.io.savemat('./output_'+str(num) + opt.netpath +'/test_index_'+str(num)+'.mat',
                 {'test_index': list(test_index)})
train_index = total_index-test_index

GLO.set_value('train_index', train_index)
GLO.set_value('test_index', test_index)

# print(GLO._global_dict)
scipy.io.savemat('./output_'+str(num) + opt.netpath +'/upper_bound_'+str(num)+'.mat',
                 {'upper_bound': GLO.get_value('upper_bound')})
scipy.io.savemat('./output_'+str(num) + opt.netpath +'/lower_bound_'+str(num)+'.mat',
                 {'lower_bound': GLO.get_value('lower_bound')})

MyDataset = datasets.process_data(input, target)
train_dataloader, test_dataloader = MyDataset.encap(num)

test_iter = iter(test_dataloader)
test_data = test_iter.next()
test_input_cpu, test_target_cpu = test_data
test_target_actual = test_target[list(test_index)]  # actual temp
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

logger = SummaryWriter()
# tensorboard --logdir=./runs/

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

# trainLogger = open('%s/train.log' % ('network_'+str(num)+ opt.netpath), 'w')
log_step_interval = 100
optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(
    opt.betas0, opt.betas1), weight_decay=opt.weight_decay)

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
        if (global_iter_num % log_step_interval == 0) or (global_iter_num == opt.epochs * len(train_dataloader)):
            # print('[%d/%d][%d/%d] %d loss: %f'
            #       % (epoch, opt.epochs, i, len(train_dataloader), global_iter_num, loss.item()))
            # sys.stdout.flush()
            # trainLogger.write('%d\t%d\t%d\t%d\t%f\n'
            #                   % (epoch, opt.epochs, i, len(train_dataloader), loss.item()))
            # trainLogger.flush()
            
            logger.add_scalar("loss/train loss", loss.item(),
                              global_step=global_iter_num)

            with torch.no_grad():
                net.eval()
                test_output, _, _ = net(test_input.float())
                test_loss = criterion(test_output.float(), test_target.float())
            logger.add_scalar("loss/test loss", test_loss.item(),
                              global_step=global_iter_num)
            logger.add_scalars("loss/loss together", {'train loss': loss.item(
            ), 'test loss': test_loss.item()}, global_step=global_iter_num)
            net.train()

            if (global_iter_num % 100 == 0) or (global_iter_num == opt.epochs * len(train_dataloader)):
                test_output = test_output[:, 0].cpu().numpy()
                test_output = utils.square2rec(test_output)  # contain denormalization
                test_output_process['test_output_' +
                                    str(global_iter_num)] = test_output                

            for name, param in net.named_parameters():
                logger.add_histogram(name, param.clone().cpu(
                ).data.numpy(), global_step=global_iter_num)

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
        scipy.io.savemat('./test_output_process_'+str(num)+ opt.netpath +'/test_output.mat',
                         {'test_output': test_output})

scipy.io.savemat('./test_output_process_'+str(num)+ opt.netpath +'/test_output_process_' +
                 str(num)+'.mat', {'test_output_process': test_output_process})
# trainLogger.close()
logger.export_scalars_to_json('./all_scalars_'+str(num)+opt.netpath+'.json')
logger.close()

# shape (100,50,34) (33,4,100,50)
test_error = abs(test_output-test_target_actual)
ele_num = np.prod([test_error.shape[-2], test_error.shape[-1]])
# print('ele_num: ', ele_num)
test_error_avg = [np.sum(test_error[i])/opt.seq_len /
                  ele_num for i in range(test_error.shape[0])]
test_error_max = [np.max(test_error[i]) for i in range(test_error.shape[0])]
print('test_error_avg: ', test_error_avg, '\n', 'max(test_error_avg): ', max(
    test_error_avg), '\n', 'max(test_error_max): ', max(test_error_max))

# trainLogger = open('%s/train.log' % ('network_'+str(num)+ opt.netpath), 'a')
# trainLogger.write('%.4f\n'%test_error_avg)
# trainLogger.close()
