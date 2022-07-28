# U-net structure + Net_rev1
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from utils import GLO


def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, dropout=False, special_stride=False, pool=False):
    seq_len = GLO.get_value('seq_len')
    dropout_val = GLO.get_value('dropout')
    slope = GLO.get_value('slope')
    block = nn.Sequential()

    if not transposed:
        if pool:
            block.add_module('%s_pool' %
                             name, nn.MaxPool2d(kernel, 1, 1))  # padding=1
        else:
            block.add_module('%s_conv' % name, nn.Conv2d(
                in_c, out_c, kernel, stride, 1, bias=False))  # padding=1
    else:
        if special_stride:
            block.add_module('%s_tconv' % name, nn.ConvTranspose2d(
                in_c, out_c, kernel, 1, 1, bias=False))
        else:
            block.add_module('%s_tconv' % name, nn.ConvTranspose2d(
                in_c, out_c, kernel, stride, 1, bias=False))  # padding=1

    if bn:
        # nn.InstanceNorm2d(out_c,affine=True)
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout:
        block.add_module('%s_dropout' % name, nn.Dropout2d(dropout, inplace=True))

    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))  # True
    else:
        block.add_module('%s_leakyrelu' %
                         name, nn.LeakyReLU(slope, inplace=True))  # True
    return block


def linear_block(in_n, out_n, name):
    slope = GLO.get_value('slope')
    block = nn.Sequential()
    block.add_module('%s_linear' % name, nn.Linear(in_n, out_n))
    block.add_module('%s_leakyrelu' %
                     name, nn.LeakyReLU(slope, inplace=True))  # True
    return block


kernel = 4
stride = 2


class net(nn.Module):
    def __init__(self, input_nc, output_nc, nf, hidden_dim, fc_dim):
        super(net, self).__init__()
        seq_len = GLO.get_value('seq_len')
        slope = GLO.get_value('slope')
        assert input_nc == seq_len, 'input_nc not equal to seq_len'
        assert output_nc == seq_len, 'input_nc not equal to seq_len'

        # 100->50
        layer_idx = 1
        name = 'layer%d' % layer_idx
        layer1 = nn.Sequential()
        layer1.add_module('%s_conv' % name, nn.Conv2d(
            input_nc, nf*seq_len, kernel, stride, 1, bias=False))
        layer1.add_module('%s_leakyrelu' %
                          name, nn.LeakyReLU(slope, inplace=True))
        # 50->25
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer2 = blockUNet(nf*seq_len, nf*2*seq_len, name,
                           transposed=False, bn=True, relu=False, dropout=False)
        # 25->24
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer3 = blockUNet(nf*2*seq_len, nf*2*seq_len, name, transposed=False,
                           bn=True, relu=False, dropout=False, pool=True)
        # 24->12
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer4 = blockUNet(nf*2*seq_len, nf*4*seq_len, name,
                           transposed=False, bn=True, relu=False, dropout=False)
        # 12->6
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer5 = blockUNet(nf*4*seq_len, nf*8*seq_len, name,
                           transposed=False, bn=True, relu=False, dropout=False)
        # 6->3
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer6 = blockUNet(nf*8*seq_len, nf*8*seq_len, name,
                           transposed=False, bn=True, relu=False, dropout=False)

        # encoder fully connected layer
        # fclayer_idx = 1
        # name = 'fclayer%d' % fclayer_idx
        # fclayer1 = linear_block(nf*8*3*3*seq_len, 2048*seq_len, name)

        # fclayer_idx += 1
        # name = 'fclayer%d' % fclayer_idx
        # fclayer2 = linear_block(2048*seq_len, 512*seq_len, name)

        # fclayer3 = nn.Linear(512*seq_len, 100*seq_len)
        fclayer_idx = 1
        name = 'fclayer%d' % fclayer_idx
        fclayer1 = linear_block(nf*8*3*3*seq_len, fc_dim*seq_len, name)

        fclayer_idx += 1
        name = 'fclayer%d' % fclayer_idx
        fclayer2 = linear_block(fc_dim*seq_len, 100*seq_len, name)

        # LSTM
        lstm = nn.LSTM(input_size=100, hidden_size=hidden_dim,
                       num_layers=1, batch_first=True)
        linear = nn.Sequential(nn.Linear(in_features=hidden_dim, out_features=100), nn.LeakyReLU(
            slope, inplace=True))  # nn.Tanh()

        # decoder fully connected layer
        # fclayer_idx += 1
        # name = 'dfclayer%d' % fclayer_idx
        # dfclayer3 = linear_block(100*seq_len, 512*seq_len, name)

        # fclayer_idx -= 1
        # name = 'dfclayer%d' % fclayer_idx
        # dfclayer2 = linear_block(512*seq_len, 2048*seq_len, name)

        # dfclayer1 = nn.Linear(2048*seq_len, nf*8*3*3*seq_len)
        fclayer_idx += 1
        name = 'dfclayer%d' % fclayer_idx
        dfclayer3 = linear_block(100*seq_len, fc_dim*seq_len, name)

        fclayer_idx -= 1
        name = 'dfclayer%d' % fclayer_idx
        dfclayer2 = linear_block(fc_dim*seq_len, nf*8*3*3*seq_len, name)

        # 3->6
        name = 'dlayer%d' % layer_idx
        dlayer6 = blockUNet(nf*8*seq_len, nf*8*seq_len, name,
                            transposed=True, bn=True, relu=True, dropout=True)
        # 6->12
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer5 = blockUNet(nf*8*seq_len*2, nf*8*seq_len, name,
                            transposed=True, bn=True, relu=True, dropout=True)
        # 12->24
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer4 = blockUNet(int(nf*8*seq_len*1.5), nf*4*seq_len, name,
                            transposed=True, bn=True, relu=True, dropout=False)
        # 24->25
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer3 = blockUNet(int(nf*4*seq_len*1.5), nf*2*seq_len, name, transposed=True,
                            bn=True, relu=True, dropout=False, special_stride=True)
        # 25->50
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer2 = blockUNet(nf*2*seq_len*2, nf*seq_len, name,
                            transposed=True, bn=True, relu=True, dropout=False)
        # 50->100
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer1 = nn.Sequential()
        dlayer1.add_module('%s_tconv' % name, nn.ConvTranspose2d(
            nf*seq_len*2, output_nc, kernel, stride, 1, bias=False))
        dlayer1.add_module('%s_tanh' % name, nn.Tanh())

        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.layer5 = layer5
        self.layer6 = layer6

        self.fclayer1 = fclayer1
        self.fclayer2 = fclayer2
        # self.fclayer3 = fclayer3

        self.lstm = lstm
        self.linear = linear

        self.dfclayer3 = dfclayer3
        self.dfclayer2 = dfclayer2
        # self.dfclayer1 = dfclayer1

        self.dlayer6 = dlayer6
        self.dlayer5 = dlayer5
        self.dlayer4 = dlayer4
        self.dlayer3 = dlayer3
        self.dlayer2 = dlayer2
        self.dlayer1 = dlayer1

        self.nf=nf

    def forward(self, x, *args):
        seq_len = GLO.get_value('seq_len')
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)

        out6 = out6.view(out6.size(0), -1)
        fc1 = self.fclayer1(out6)
        fc2 = self.fclayer2(fc1)
        # fc3=self.fclayer3(fc2)
        # fc3=fc3.view(-1,seq_len,100)
        fc2 = fc2.view(-1, seq_len, 100)

        if args:
            h_0, c_0 = args[0], args[1]
            recurrent_features, (hn, cn) = self.lstm(fc2, (h_0, c_0))
        else:
            recurrent_features, (hn, cn) = self.lstm(fc2)

        outputs = self.linear(recurrent_features)
        outputs = outputs.view(outputs.shape[0], -1)

        dfc3 = self.dfclayer3(outputs)
        dfc2 = self.dfclayer2(dfc3)
        # dfc1=self.dfclayer1(dfc2)
        # dfc1=torch.reshape(dfc1,(dfc1.size(0),self.nf*8*seq_len,3,3))
        dfc2 = torch.reshape(dfc2, (dfc2.size(0), self.nf*8*seq_len, 3, 3))

        dout6 = self.dlayer6(dfc2)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2 = torch.cat([dout3, out2], 1)
        dout2 = self.dlayer2(dout3_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)

        return dout1, hn, cn
