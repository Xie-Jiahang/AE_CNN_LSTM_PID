import scipy.io
import numpy as np

from utils import GLO


class error_predict():
    def __init__(self, delta_t, steps, netpath):
        self.delta_t = delta_t
        self.batch_size = GLO.get_value('batch_size')
        self.seq_len = GLO.get_value('seq_len')
        self.num = GLO.get_value('num')
        self.steps = steps
        self.traceback = 1
        self.tau = 1
        error_with_target = scipy.io.loadmat(
            './output_'+str(self.num)+ netpath +'/error_with_target.mat')
        self.error_with_target = error_with_target['error_with_target']

    # def fitting(self):
    #     for seq in range(self.seq_len):
    #         error_with_target=self.error_with_target[:,:,:,seq]

    #         y=np.zeros((np.prod(error_with_target.shape[-2:]),self.delta_t*(self.steps-self.traceback)))
    #         x=np.zeros_like(y)
    #         for i in range(self.delta_t):
    #             for j in range(self.steps-self.traceback):
    #                 idx1=i*(self.steps-self.traceback)+j
    #                 idx2=j+self.traceback

    #                 y[:,idx1]=np.reshape(error_with_target[i,idx2],(-1,))
    #                 x[:,idx1]=np.reshape(error_with_target[i,idx2-1],(-1,))

    #         # do linear regression for each element in matrix -- 100*50 regression models
    #         A=np.dot(y,np.linalg.pinv(x))
    #         C=y-np.dot(A,x)
    #         C=np.mean(C,axis=1)

    #     return A,C

    def fitting(self):
        A = {}
        C = {seq: 0 for seq in range(self.seq_len)}
        for seq in range(self.seq_len):
            error_with_target = self.error_with_target[:, :, :, seq]

            y = np.zeros((np.prod(
                error_with_target.shape[-2:]), self.delta_t*(self.steps-(self.traceback+self.tau))))
            x = np.zeros((self.traceback+self.tau, np.prod(
                error_with_target.shape[-2:]), self.delta_t*(self.steps-(self.traceback+self.tau))))
            for i in range(self.delta_t):
                for j in range(self.steps-(self.traceback+self.tau)):
                    idx1 = i*(self.steps-(self.traceback+self.tau))+j
                    idx2 = j+self.traceback+self.tau

                    y[:, idx1] = np.reshape(error_with_target[i, idx2], (-1,))
                    for k in range(self.traceback+self.tau):
                        x[k, :, idx1] = np.reshape(
                            error_with_target[i, idx2-1-k], (-1,))

            for k in range(self.traceback+self.tau):
                # do linear regression for each element in matrix -- 100*50 regression models
                A[(seq, k)] = np.dot(y, np.linalg.pinv(x[k])) / \
                    (self.traceback+self.tau)
                C[seq] += (y-np.dot(A[(seq, k)], x[k])) / \
                    (self.traceback+self.tau)

        return A, C

    # def predict(self,A,C): # when the real data at t come, need to redo regression???
    #     C=np.expand_dims(C,-1)
    #     all_error_result=np.zeros((self.steps,self.batch_size,self.seq_len,self.error_with_target.shape[-2],self.error_with_target.shape[-1]))

    #     for seq in range(self.seq_len):
    #         error_with_target=self.error_with_target[:,:,:,seq]

    #         error_result=np.zeros((self.steps,self.batch_size,error_with_target.shape[-2],error_with_target.shape[-1]))
    #         # 1st dim--delta_t; 2nd dim--steps
    #         for i in range(self.traceback):
    #             error_result[i]=np.mean(error_with_target[:,i],axis=0)

    #         # x=np.zeros((np.prod(error_with_target.shape[-2:]),1))
    #         for step in range(self.traceback,self.steps):
    #             x=np.reshape(error_result[step-1],(-1,1))
    #             y=np.dot(A,x)+C
    #             error_result[step,:]=np.reshape(y,(error_with_target.shape[-2],error_with_target.shape[-1]))

    #         all_error_result[:,:,seq]=error_result
    #     return all_error_result

    def predict(self, A, C):  # when the real data at t come, need to redo regression???
        all_error_result = np.zeros((self.steps, self.batch_size, self.seq_len,
                                    self.error_with_target.shape[-2], self.error_with_target.shape[-1]))

        for seq in range(self.seq_len):
            # C[seq]=np.expand_dims(C[seq],-1)
            C[seq] = np.mean(C[seq], axis=1)
            error_with_target = self.error_with_target[:, :, :, seq]

            error_result = np.zeros(
                (self.steps, self.batch_size, error_with_target.shape[-2], error_with_target.shape[-1]))
            # 1st dim--delta_t; 2nd dim--steps
            for i in range(self.traceback+self.tau):
                error_result[i] = np.mean(error_with_target[:, i], axis=0) # np.mean(error_with_target[:, -i], axis=0)

            # x=np.zeros((np.prod(error_with_target.shape[-2:]),1))
            y = np.zeros((np.prod(error_with_target.shape[-2:]),))
            for step in range(self.traceback+self.tau, self.steps):
                for k in range(self.traceback+self.tau):
                    x = np.reshape(error_result[step-1-k], (-1,))
                    y += np.dot(A[(seq, k)], x)+C[seq]
                error_result[step, :] = np.reshape(
                    y, (error_with_target.shape[-2], error_with_target.shape[-1]))

            all_error_result[:, :, seq] = error_result
        return all_error_result
