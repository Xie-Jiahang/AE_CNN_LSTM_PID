import torch
import os
import scipy.io
import numpy as np


class GLO():
    _global_dict = {'num': 0, 'fill_value': 0, 'mean': 0, 'std': 0, 'upper_bound': 0,
    'lower_bound': 0, 'batch_size': 0, 'seq_len': 4, 'train_index': 0, 'test_index': 0,
    'dropout': 0.5, 'slope': 0.2, 'hidden_dim': 128}

    @staticmethod
    def get_value(key, defValue=None):
        try:
            return GLO._global_dict[key]
        except KeyError:
            return defValue

    @staticmethod
    def set_value(key, value):
        GLO._global_dict[key] = value


def prepare_sensor(data, num, netpath):
    # fill value --> (mean of the sensor data each) for each input; sensor area --> "measured" data
    # 100*50*N    
    fill_value = [0]*data.shape[-1]

    sensor = np.zeros_like(data)
    sensor_x = 50
    sensor_y = 25

    for i in range(data.shape[-1]):
        sensor_area1 = data[sensor_x-1-25:sensor_x +
                            2-25, sensor_y-1:sensor_y+2, i]
        sensor_area2 = data[sensor_x-1:sensor_x +
                            2, sensor_y-1:sensor_y+2, i]
        sensor_area3 = data[sensor_x-1+25:sensor_x +
                            2+25, sensor_y-1:sensor_y+2, i]
        fill_value[i] = np.mean(np.concatenate(
            (sensor_area1, sensor_area2, sensor_area3)))

        tmp = fill_value[i]*np.ones_like(data[:, :, i])
        tmp[sensor_x-1-25:sensor_x+2-25,
            sensor_y-1:sensor_y + 2] = sensor_area1
        tmp[sensor_x-1:sensor_x+2, sensor_y-1:sensor_y + 2] = sensor_area2
        tmp[sensor_x-1+25:sensor_x+2+25,
            sensor_y-1:sensor_y + 2] = sensor_area3
        sensor[:, :, i] = tmp

    GLO.set_value('fill_value', fill_value)
    if os.path.isdir('./output_'+str(num) + netpath) == False:
        os.makedirs('./output_'+str(num) + netpath)
    scipy.io.savemat('./output_'+str(num) + netpath +'/sensor_2D_'+str(num) +
                        '.mat', {'sensor': sensor})

    print('sensor data created!')


def save_adjusted(__in, __target, num, netpath): # save normalized 100*100 data
    _in_square = rec2square(__in)
    _target_square = rec2square(__target)
    _in_square = normalization(_in_square)
    _target_square = normalization(_target_square)    
    scipy.io.savemat('./output_'+str(num) + netpath +'/adjusted_input_'+str(num)+'.mat',
                        {'adjusted_input': _in_square})
    scipy.io.savemat('./output_'+str(num) + netpath +'/adjusted_target_'+str(num)+'.mat',
                        {'adjusted_target': _target_square})

def normalization(__square):  # for each point/pixel
    # N*100*100
    _upper_bound = np.max(__square, axis=0)
    _lower_bound = np.min(__square, axis=0)
    __square = (__square-_lower_bound)/(_upper_bound-_lower_bound)
    # bounds are set by _target_square
    GLO.set_value('upper_bound', _upper_bound)
    GLO.set_value('lower_bound', _lower_bound)
    return __square


def form_time_seq(original, seq_len):
    seq = []
    for i in range(len(original)-seq_len+1):
        seq.append(original[i:i+seq_len])
    return seq


def rec2square(data):  # concatenate rec to square
    fill_value = GLO.get_value('fill_value')
    upper_bound = GLO.get_value('upper_bound')
    lower_bound = GLO.get_value('lower_bound')
    mean = GLO.get_value('mean')

    _cat = np.concatenate((data, data), -1)    
    return _cat


def square2rec(data):  # batch num is the last dimension
    _upper_bound = GLO.get_value('upper_bound')
    _lower_bound = GLO.get_value('lower_bound')
    seq_len = GLO.get_value('seq_len')
    test_index = GLO.get_value('test_index')

    _res = data*(_upper_bound-_lower_bound)+_lower_bound  # denormalization    

    if len(data.shape) == 3:
        _res = (_res[:, :, :50]+_res[:, :, 50:])/2
    elif len(data.shape) == 4:
        _res = (_res[:, :, :, :50]+_res[:, :, :, 50:])/2
    elif len(data.shape) == 5:
        _res = (_res[:, :, :, :, :50]+_res[:, :, :, :, 50:])/2
    return _res


def error_square2rec(error, denormalize):  # batch num is the first dimension
    upper_bound = GLO.get_value('upper_bound')
    lower_bound = GLO.get_value('lower_bound')
    std = GLO.get_value('std')

    if denormalize == 1:  # input error needs denormalization
        _res = error*(upper_bound-lower_bound)  # denormalization        
    else:
        _res = error    

    if len(error.shape) == 4:
        _res = (_res[:, :, :, :50]+_res[:, :, :, 50:])/2
    elif len(error.shape) == 5:
        _res = (_res[:, :, :, :, :50]+_res[:, :, :, :, 50:])/2
    return _res


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:  # InstanceNorm
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


# test for 1 image, normalized value
def test_network_sensitivity(input, test_pid, target, netG):
    y_t = torch.from_numpy(target[test_pid]).cuda()

    s = torch.from_numpy(input[test_pid]).cuda()
    s = s.unsqueeze(0)
    s = s.unsqueeze(0)
    output_original = netG(s.float())
    error_original = y_t-output_original    

    levels = [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, -
              1e1, -1e0, -1e-1, -1e-2, -1e-3, -1e-4]
    result = np.zeros((len(levels), 5))
    for i in range(len(levels)):
        delta_s = levels[i]*torch.ones(s.shape[-2], s.shape[-1]).cuda()
        output = netG((s+delta_s).float())
        error = y_t-output        

        delta_y = output-output_original        
        result[i, 0] = levels[i]
        result[i, 1] = torch.max(delta_y).item()
        result[i, 2] = torch.min(delta_y).item()
        result[i, 3] = torch.norm(delta_y).item()/torch.norm(delta_s).item()
        result[i, 4] = torch.norm(delta_y, p=1).item() / \
            torch.norm(delta_s, p=1).item()

    np.savetxt('./output/network_sensitivity.txt', result, fmt='%.05f')

    with open('./output/network_sensitivity.txt', 'a') as f:
        f.write(
            'delta_s    delta_y(max)    delta_y(min)    Frobenius norm    p norm'+'\n')


def prepare_data(num, netpath): # load upper & lower bound matrices
    num = GLO.get_value('num')
    upper_bound = scipy.io.loadmat('./output_'+str(num) + netpath +'/upper_bound_'+str(num)+'.mat')
    upper_bound = upper_bound['upper_bound']
    lower_bound = scipy.io.loadmat('./output_'+str(num) + netpath +'/lower_bound_'+str(num)+'.mat')
    lower_bound = lower_bound['lower_bound']
    GLO.set_value('upper_bound', upper_bound)
    GLO.set_value('lower_bound', lower_bound)    
