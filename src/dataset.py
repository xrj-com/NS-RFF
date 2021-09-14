import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from marveltoolbox.utils import TorchComplex as tc
from .preprocessing import main
import os

class RFdataset(torch.utils.data.Dataset):
    def __init__(self, device_ids, test_ids, flag='data_new', SNR=None, rand_max_SNR=None, is_return_more=False):
        if len(device_ids)> 1:
            device_flag = '{}-{}'.format(device_ids[0], device_ids[-1])
        else:
            device_flag = str(device_ids[0])
        test_flag = '-'.join([str(i) for i in test_ids])
        file_name = '{}_dv{}_id{}.pth'.format(flag, device_flag, test_flag)
        file_name = '/workspace/DATASET/ZigBee/new/{}'.format(file_name)
        if not os.path.isfile(file_name):
            main(device_ids, test_ids, flag=flag)
     
        self.data = torch.load(file_name)
        self.snr = SNR
        self.max_snr = rand_max_SNR
        self.is_return_more = is_return_more

        
    def __getitem__(self, index):
        idx = self.data['idx'][index]
        x        = self.data['x'][index][idx:idx+1280, :].view(1, -1, 2)
        x_origin = self.data['x_origin'][index][idx:idx+1280, :].view(1, -1, 2)
        x_fo     = self.data['x_fo'][index][idx:idx+1280, :].view(1, -1, 2)
        x_fopo   = self.data['x_fopo'][index][idx:idx+1280, :].view(1, -1, 2)
        y        = self.data['y'][index]
        length   = self.data['length'][index]
        coarse_freq = self.data['coarse_freq'][index]
        fine_freq = self.data['fine_freq'][index]
        phase = self.data['phase'][index]
        # rec_freq = (rec_freq + 142000)/(142000-23000)
        if not self.snr is None:
            x_origin += tc.awgn(x_origin, self.snr, SNR_x=30)
            x_fopo += tc.awgn(x_fopo, self.snr, SNR_x=30)
            x += tc.awgn(x, self.snr, SNR_x=30)
        
        if not self.max_snr is None:
            rand_snr = torch.randint(5, self.max_snr, (1,)).item()
            x_origin += tc.awgn(x_origin, rand_snr, SNR_x=30)
            x_fopo += tc.awgn(x_fopo, rand_snr, SNR_x=30)
            x += tc.awgn(x, rand_snr, SNR_x=30)
        if self.is_return_more:
            return x_origin, y, x, x_fo, x_fopo, coarse_freq, fine_freq, phase
        else:
            return x_origin, y, x, x_fo, x_fopo

    def __len__(self):
        return len(self.data['y'])



if __name__ == "__main__":
    test = RFdataset(device_ids=range(45), test_ids=[1,2,3,4], flag='data_new')
    print(len(test))
    print(test[0][0].shape)
    # min_freq = 1000000
    # max_freq = -1000000
    # sum_freq = 0.0
    # for i in range(len(test)):
    #     freq = test[i][0]
    #     if freq.max() > max_freq:
    #         max_freq = freq.max()
    #     if freq.min() < min_freq:
    #         min_freq = freq.min()
    #     sum_freq += freq.mean()
    # print(max_freq)
    # print(min_freq)
    # print(sum_freq/len(test))