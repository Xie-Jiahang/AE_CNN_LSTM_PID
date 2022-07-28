import numpy as np
import scipy.io
from sklearn.linear_model import LinearRegression

import pid
from utils import GLO


class error_predict():
    def __init__(self, delta_t, steps, netpath):
        self.delta_t = delta_t
        self.batch_size = GLO.get_value('batch_size')
        self.seq_len = GLO.get_value('seq_len')
        self.num = GLO.get_value('num')
        self.steps = steps
        self.traceback = 3
        self.tau = 4
        error_with_target = scipy.io.loadmat(
            './output_'+str(self.num)+ netpath +'/error_with_target.mat')
        self.error_with_target = error_with_target['error_with_target']

    def fitting(self):
        all_reg_model = []
        for seq in range(self.seq_len):
            error_with_target = self.error_with_target[:, :, :, seq]

            y = np.zeros((self.delta_t*(self.steps-(self.traceback+self.tau)),
                         error_with_target.shape[-2], error_with_target.shape[-1]))
            x = np.zeros((self.delta_t*(self.steps-(self.traceback+self.tau)), 2 *
                         self.traceback, error_with_target.shape[-2], error_with_target.shape[-1]))
            for i in range(self.delta_t):
                for j in range(self.steps-(self.traceback+self.tau)):  # 0~92
                    idx1 = i*(self.steps-(self.traceback+self.tau))+j
                    idx2 = j+self.traceback+self.tau  # 7~99

                    y[idx1] = error_with_target[i, idx2]
                    x[idx1, 0] = error_with_target[i, idx2-1]
                    x[idx1, 1] = error_with_target[i, idx2-2]
                    x[idx1, 2] = error_with_target[i, idx2-3]
                    x[idx1, 3] = error_with_target[i, idx2-1-self.tau]
                    x[idx1, 4] = error_with_target[i, idx2-2-self.tau]
                    x[idx1, 5] = error_with_target[i, idx2-3-self.tau]

            # do linear regression for each element in matrix -- 100*50 regression models
            reg_model = []
            for i in range(error_with_target.shape[-2]):
                for j in range(error_with_target.shape[-1]):
                    Y = y[:, i, j]
                    X = x[:, :, i, j]
                    reg = LinearRegression().fit(X, Y)
                    reg_model.append(reg)

            all_reg_model.append(reg_model)
        return all_reg_model

    def predict(self, all_reg_model):  # when the real data at t come, need to redo regression???
        all_error_result = np.zeros((self.steps, self.batch_size, self.seq_len,
                                    self.error_with_target.shape[-2], self.error_with_target.shape[-1]))

        for seq in range(self.seq_len):
            error_with_target = self.error_with_target[:, :, :, seq]
            reg_model = all_reg_model[seq]

            error_result = np.zeros(
                (self.steps, 1, error_with_target.shape[-2], error_with_target.shape[-1]))
            # 1st dim--delta_t(samples); 2nd dim--steps
            for i in range(self.traceback+self.tau):
                error_result[i] = np.mean(error_with_target[:, i], axis=0) # np.mean(error_with_target[:, -i], axis=0)

            x = np.zeros(
                (2*self.traceback, error_with_target.shape[-2], error_with_target.shape[-1]))
            for step in range(self.traceback+self.tau, self.steps):
                x[0] = error_result[step-1]
                x[1] = error_result[step-2]
                x[2] = error_result[step-3]
                x[3] = error_result[step-1-self.tau]
                x[4] = error_result[step-2-self.tau]
                x[5] = error_result[step-3-self.tau]

                for i in range(error_with_target.shape[-2]):
                    for j in range(error_with_target.shape[-1]):
                        X = np.array([x[:, i, j]])
                        error_result[step, :, i, j] = reg_model[i *
                                                                error_with_target.shape[-1]+j].predict(X)

            all_error_result[:, :, seq] = error_result
        return all_error_result
