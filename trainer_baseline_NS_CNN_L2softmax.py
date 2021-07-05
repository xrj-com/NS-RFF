from src.models import *
from src.trainer import RFFTrainer, RFFConfs

class Confs(RFFConfs):
    def __init__(self, train_snr=None, device=0, d1=None, d2=None, z_dim=512):
        super().__init__(train_snr, device, d1, d2, z_dim)
        
    def get_flag(self):
        self.eval_model = NS_CLF_L2Softmax
        self.data_idx = 0
        self.flag = 'Baseline-NS-CNN-L2softmax-snr{}-d1={}-d2={}-nz={}'.format(self.train_snr, self.d1, self.d2, self.z_dim)

class Trainer(RFFTrainer, Confs):
    def __init__(self, train_snr=None, device=0, d1=8, d2=24, z_dim=512):
        Confs.__init__(self, train_snr, device, d1, d2, z_dim)
        RFFTrainer.__init__(self)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.run(load_best=True, retrain=False, is_del_loger=False)