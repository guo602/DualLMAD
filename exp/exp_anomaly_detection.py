from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment,get_best_f1_upper
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from tqdm import tqdm
warnings.filterwarnings('ignore')
from pathlib import Path

class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args, setting):
        super(Exp_Anomaly_Detection, self).__init__(args)
        self.out_dir = Path("output") / setting
        self.out_dir.mkdir(exist_ok=True, parents=True)
        self.board_internal = 10

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    def _select_scheduler(self, model_optim, T_max):
        return torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)

    def vali(self, vali_data, vali_loader, criterion, writer, stage, epoch):
        board_loss_list = []
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _, batch_x_mark) in enumerate(tqdm(vali_loader)):
                board_step = epoch * len(vali_loader) + i
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
                board_loss_list.append(loss.item())
                if board_step % self.board_internal == 0:
                    writer.add_scalar(tag=f"stage{stage}_vali_loss", scalar_value=np.mean(board_loss_list), global_step=board_step)
                    board_loss_list = []

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting, writer, stage):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        scheduler = self._select_scheduler(model_optim, 10)
        criterion = self._select_criterion()
        board_loss_list = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark) in enumerate(tqdm(train_loader)):
                board_step = epoch * len(train_loader) + i
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                
                outputs = self.model(batch_x, batch_x_mark, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())
                board_loss_list.append(loss.item())
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                if board_step % self.board_internal == 0:
                    writer.add_scalar(tag=f"stage{stage}_train_loss", scalar_value=np.mean(board_loss_list), global_step=board_step)
                    board_loss_list = []


                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, writer, stage, epoch)
            # test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if self.args.cos:
                scheduler.step()
            # adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        # if test:
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        attens_energy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y,batch_x_mark) in enumerate(tqdm(train_loader)):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs = self.model(batch_x, None, None, None)
                # criterion
                # score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = self.anomaly_criterion(batch_x, outputs)

                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        # torch.cuda.empty_cache()

        attens_energy = []
        test_labels = []
        outputs_list = []
        raw_data = []
        for i, (batch_x, batch_y, batch_x_mark) in enumerate(tqdm(test_loader)):
            batch_x = batch_x.float().to(self.device)
            # batch_x_mark = batch_x_mark.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x, None, None, None)
            # criterion
            score = self.anomaly_criterion(batch_x, outputs)
            score = score.detach().cpu().numpy()
            raw_data.append(batch_x[:, -1].detach().cpu().numpy())
            # attens_energy.append(score[:, -1])
            # test_labels.append(batch_y[:, -1])
            attens_energy.append(score)
            test_labels.append(batch_y)
            outputs_list.append(outputs[:, -1, :].detach().cpu().numpy())

        attens_energy = np.concatenate(attens_energy, axis=0)
        test_energy = np.array(attens_energy)
        outputs_array = np.concatenate(outputs_list, axis=0)
        raw_data = np.concatenate(raw_data, axis=0)
        
        
        # (3) evaluation on the test set
        test_labels = np.concatenate(test_labels, axis=0)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)
        print(f"train energy:{train_energy.shape} test_energy: {test_energy.shape} test_labels: {gt.shape}")


        np.save(self.out_dir / "train_score.npy", train_energy)
        np.save(self.out_dir / "test_score.npy", test_energy)
        np.save(self.out_dir / "label.npy", gt)
        np.save(self.out_dir / "recon.npy", outputs_array)
        np.save(self.out_dir / "raw_data.npy", raw_data)

        label = test_labels.reshape(-1)
        score = np.mean(test_energy,axis = -1).reshape(-1)
        results,_ = get_best_f1_upper(score,label)
        print(f"f1:{results[0]} p:{results[1]} r:{results[2]}")

        
