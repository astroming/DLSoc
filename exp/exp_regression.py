from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.metrics import evaluate
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
warnings.filterwarnings('ignore')



class Exp_Regression(Exp_Basic):
    def __init__(self, args):
        super(Exp_Regression, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        # test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = train_data.max_seq_len
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.static_in = train_data.df_static.shape[1]
        self.args.tasks = train_data.df_task['task'].unique()
        # self.args.num_class = len(train_data.class_names)
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        # if self.args.use_multi_gpu and self.args.use_gpu:
        #     model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        # model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self,loss_name='MSE'):
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'L1':
            return nn.L1Loss()
        elif loss_name == 'MAPE':
            return mape_loss()
        elif loss_name == 'MASE':
            return mase_loss()
        elif loss_name == 'SMAPE':
            return smape_loss()

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')
        # test_data, test_loader = self._get_data(flag='TEST')

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True,delta=self.args.early_delta)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            epoch_time = time.time()
            iter_count = 0

            train_loss = []
            preds = []
            trues = []
            self.model.train()
            for i, (batch_x, static_x, batch_task, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                static_x = static_x.float().to(self.device)
                batch_task = batch_task.numpy()#to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x, static_x, batch_task)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20.0)
                model_optim.step()

                pred = outputs.detach().cpu()
                preds.append(pred)
                trues.append(batch_y.detach().cpu())

            train_loss = np.average(train_loss)

            preds = torch.cat(preds, 0).flatten().numpy()
            trues = torch.cat(trues, 0).flatten().numpy()
            preds = train_data.scaler_l.inverse_transform(preds.reshape(preds.shape[0], -1))
            trues = train_data.scaler_l.inverse_transform(trues.reshape(trues.shape[0], -1))

            train_score = evaluate(trues, preds)
            vali_loss, val_score = self.vali(vali_data, vali_loader, criterion)
            # test_loss, test_score = self.vali(test_data, test_loader, criterion)

            early_stopping(-1*val_score[self.args.evaluation], self.model, self.args, train_score,val_score,epoch)

            log_str = f"--------------- CV: {self.args.cv_id} --- Epoch: {epoch} --- cost time: {time.time() - epoch_time} ---------------\n"+f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Vali Loss: {vali_loss:.4f}\n"+\
                  f" train score: {train_score}\n"+    f" valid score: {val_score}\n"+\
                  f" Best valid score: {early_stopping.best_val_score} Best epoch: {early_stopping.best_epoch}"
            # print(f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Vali Loss: {vali_loss:.4f}\n"
            #       f" train score: {train_score}\n"
            #       f" valid score: {val_score}\n"
            #       f" Best valid score: {early_stopping.best_val_score} Best epoch: {early_stopping.best_epoch}")
            log_file = os.path.join(self.args.save_path, 'log.log')
            with open(log_file, 'a') as f:
                print(log_str, file=f)
                print(log_str)

            if early_stopping.early_stop:
                print("Early stopping")
                return early_stopping
                break
            if (epoch + 1) % self.args.lr_adjust_epochs == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        # best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))

        return early_stopping

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, static_x, batch_task, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                static_x = static_x.float().to(self.device)
                batch_task = batch_task.numpy()#.to(self.device)
                batch_y = batch_y.float()#.to(self.device)

                outputs = self.model(batch_x, static_x, batch_task)

                pred = outputs.detach().cpu()
                loss = criterion(pred, batch_y)
                total_loss.append(loss)

                preds.append(pred)
                trues.append(batch_y)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0).flatten().numpy()
        trues = torch.cat(trues, 0).flatten().numpy()
        preds = vali_data.scaler_l.inverse_transform(preds.reshape(preds.shape[0], -1))
        trues = vali_data.scaler_l.inverse_transform(trues.reshape(trues.shape[0], -1))

        score = evaluate(trues,preds)

        self.model.train()
        return total_loss, score

    # def test(self, setting, test=0):
    #     test_data, test_loader = self._get_data(flag='TEST')
    #     if test:
    #         print('loading model')
    #         self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
    #
    #     preds = []
    #     trues = []
    #     folder_path = './test_results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    #
    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, padding_mask) in enumerate(test_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             padding_mask = padding_mask.float().to(self.device)
    #             batch_y = batch_y.to(self.device)
    #
    #             outputs = self.model(batch_x, padding_mask, None, None)
    #
    #             preds.append(outputs.detach())
    #             trues.append(batch_y)
    #
    #     preds = torch.cat(preds, 0)
    #     trues = torch.cat(trues, 0)
    #     print('test shape:', preds.shape, trues.shape)
    #
    #     probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
    #     predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
    #     trues = trues.flatten().cpu().numpy()
    #     accuracy = cal_accuracy(predictions, trues)
    #
    #     # result save
    #     folder_path = './results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    #
    #     print('accuracy:{}'.format(accuracy))
    #     file_name='result_classification.txt'
    #     f = open(os.path.join(folder_path,file_name), 'a')
    #     f.write(setting + "  \n")
    #     f.write('accuracy:{}'.format(accuracy))
    #     f.write('\n')
    #     f.write('\n')
    #     f.close()
    #     return
