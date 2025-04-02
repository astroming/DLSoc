import argparse, os
import torch, random
import numpy as np
import pandas as pd
from shutil import copy
from exp.exp_regression import Exp_Regression
from data_provider.data_preprocess import rangelands_process

if __name__ == '__main__':
    fix_seed = 2021
    trochseed = 48
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Rangelands')

    # basic config
    parser.add_argument('--task_name', type=str, default='regression')
    parser.add_argument('--is_training', type=int,  default=1, help='status')
    parser.add_argument('--model', type=str, default='Transformer',
                        help='model name, options: [CNN, LSTM, GRU, Transformer, Mamba]')
    parser.add_argument('--predtarget', type=str, default='esm_mean_soc_pcnt')

    # data loader
    # parser.add_argument('--data', type=str,  default='UEA', help='dataset type')
    parser.add_argument('--process_rawdata', type=bool, default=True, help='if process rawdata or use processed.pkl')
    parser.add_argument('--ts_start', type=str, default='2016-01-01', help='time series start date')
    parser.add_argument('--ts_end', type=str, default='2022-12-01', help='climate end 202212')
    parser.add_argument('--ts_interpolate_method', type=str, default='linear',
                        help='linear, nearest’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘spline’, etc.')
    parser.add_argument('--processed_path', type=str, default='./dataset/rangelands/processed.pkl')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints', help="location of model checkpoints. if it is './checkpoints', the path will be model settings, else set e.g., './checkpoints/test1'")
    parser.add_argument('--save_model', type=bool, default=False, help='save best model')

    # model define
    ## basic model setting
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=10, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=1, help='num of time series layers')
    # parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')

    ## settings of Conv1d in encoder
    parser.add_argument('--d_enconv', type=int, default=128, help='dimension of Conv1d in encoder')

    ## mamba setting
    parser.add_argument('--expand', type=int, default=1, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=3, help='conv kernel size for Mamba')
    parser.add_argument('--d_ff', type=int, default=128, help='conv kernel size for Mamba')

    ## segRNN setting
    parser.add_argument('--seg_len', type=int, default=6,
                        help='the length of segmen-wise iteration of SegRNN, must be less than channels of ts')

    ## static feature encoder layer
    parser.add_argument('--static_mlp_layers', type=int, default=3, help='num of static mlp layers')
    parser.add_argument('--static_mlp_d', type=int, default=64, help='dimension of static_mlp')

    parser.add_argument('--mutlitask', type=bool, default=True, help='if use multitask')
    ## non-multitask learning
    parser.add_argument('--mlp_layers', type=int, default=3, help='num of last mlp layers')
    parser.add_argument('--mlp_layers_d', type=int, default=256, help='dimension of last mlp layers')

    ## multitask learning
    parser.add_argument('--mlp_layers_shared1', type=int, default=2, help='num of last mlp layers')
    parser.add_argument('--mlp_layers_shared2', type=int, default=1, help='num of last mlp layers')

    parser.add_argument('--mlp_layers_task1', type=int, default=1, help='num of last mlp layers')
    parser.add_argument('--mlp_layers_task2', type=int, default=1, help='num of last mlp layers')


    parser.add_argument('--dropout', type=float, default=0.02, help='dropout')
    parser.add_argument('--dropout_ts', type=float, default=0.02, help='dropout')
    parser.add_argument('--dropout_static', type=float, default=0.02, help='dropout')
    parser.add_argument('--dropout_lastmlp', type=float, default=0.02, help='dropout')

    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--activation', type=str, default='relu', help='EncoderLayer for conv gelu')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='optimizer learning rate')
    parser.add_argument('--lr_adjust_epochs', type=int, default=10, help='data loader num workers')
    parser.add_argument('--num_workers', type=int, default=14, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--early_delta', type=float, default=0.00001, help='early stopping patience')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='MAPE', help='loss function: MSE, MAPE, MASE, SMAPE, L1')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--evaluation', type=str, default='R2', help='MAE, MSE,RMSE,R2')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    parser.add_argument('--cv_seed', type=int, default=42, help='cross validation seed')
    parser.add_argument('--cv_folders', type=int, default=5, help='cross validation folders')
    parser.add_argument('--cv_id', type=int, default=1, help='must <=cv_folders-1. the i-th folder of cv')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False


    # for trochseed in range(100):
    if True:
        setting = (f'mt_{int(args.mutlitask)}_ts_start{args.ts_start}_ts_end{args.ts_end}'+
                   f'_{args.model}_torchseed{trochseed}'+
                   f'_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_enconv{args.d_enconv}' +
                   f'_mlp_layers_d{args.mlp_layers_d}_static_mlp_d{args.static_mlp_d}'+
                   f'_cv{args.cv_folders}_batch{args.batch_size}'+
                   f'_dropout_ts{args.dropout_ts}_dropout_static{args.dropout_static}_dropout_lastmlp{args.dropout_lastmlp}')
        if args.checkpoints!='./checkpoints':
            save_path = args.checkpoints
        else:
            save_path = os.path.join(args.checkpoints, setting)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        args.save_path = save_path
        
        print (args)

        for file in ['run.py']:
            copy(file, args.save_path)
        for dir in ['data_provider', 'exp', 'layers', 'models', 'utils']:
            for f in os.listdir(dir):
                if os.path.isfile(os.path.join(dir, f)):
                    dst = os.path.join(args.save_path,dir)
                    if not os.path.exists(dst):
                        os.makedirs(dst)
                    copy(os.path.join(dir, f), os.path.join(dst, f))

        'nbart_coastal_aerosol', 'nbart_blue',
        'nbart_green', 'nbart_red', 'nbart_red_edge_1', 'nbart_red_edge_2',
        'nbart_red_edge_3', 'nbart_nir_1', 'nbart_swir_2', 'nbart_swir_3', 'nbart_nir_2'
        ts_bands = ['nbart_coastal_aerosol', 'nbart_blue',
                    'nbart_green', 'nbart_red', 'nbart_red_edge_1', 'nbart_red_edge_2',
                    'nbart_red_edge_3', 'nbart_nir_1', 'nbart_swir_2', 'nbart_swir_3', 'nbart_nir_2',
                    'climate_evap', 'climate_pw', 'climate_rain', 'climate_srad', 'climate_tavg',
                    'climate_tmax', 'climate_tmin', 'climate_vpd'
                    ]

        static_f = ['radmap_v4_2019_filtered_ML_kThU.tif|band_1',
                    'radmap_v4_2019_filtered_ML_kThU.tif|band_2', 'radmap_v4_2019_filtered_ML_kThU.tif|band_3',
                    'slope_relief_class_3s.tif|band_1',
                    'topographicWetnessIndex1s.tif|band_1', 'waterDeficitTotal.tif|band_1',
                    'precipitationTotal.tif|band_1',
                    'srtm-1sec-demh-v1-COG|band_1', 'mrvbf_int|band_1', 'aspect_1s|band_1',
                    'focalrange300m_1s|band_1', 'mrrtf6g-a5_1s|band_1',
                    'plan_curvature_1s|band_1', 'PrescottIndex_01_1s_lzw|band_1',
                    'profile_curvature_1s|band_1', 'slopedeg_1s|band_1',
                    'slopepct1s|band_1', 'slope_relief|band_1', 'twi_1s(wetness)|band_1'
                    ]
        args.ts_bands=ts_bands
        args.static_f=static_f

        ##############################################
        ## save print to log file
        log_file = os.path.join(args.save_path, 'log.log')
        with open(log_file, 'w') as f:
            print(str(args), file=f)
        #######################################
        ## process raw data to time series format
        if args.process_rawdata:
            rangelands_process(args)

        ## train the model

        if args.is_training:
            for ii in range(args.itr):
                # setting record of experiments
                for cv in range(args.cv_folders):
                    # if cv==1:
                    args.cv_id = cv
                    torch.manual_seed(0)
                    torch.manual_seed(trochseed)

                    exp = Exp_Regression(args)

                    print(f'>>>>>>>start training CV: {args.cv_id}>>>>>>>>>>>>>>>>>>>>>>>>>>')
                    early_stopping = exp.train(setting)

                    def func_scores_df(cv, early_stopping):
                        scores_df = pd.DataFrame({'Model': [args.model], 'CV': cv, 'Best epoch': early_stopping.best_epoch})
                        for key in early_stopping.best_val_score.keys():
                            scores_df['val_' + key] = early_stopping.best_val_score[key]
                        for key in early_stopping.best_train_score.keys():
                            scores_df['train_' + key] = early_stopping.best_train_score[key]
                        return scores_df

                    if cv ==0:
                        scores = func_scores_df(cv, early_stopping)
                    else:
                        scores_temp = func_scores_df(cv, early_stopping)
                        scores = pd.concat([scores, scores_temp])

                ## mean and std from cvs #############
                def func_mean_std(temp, scores, describe):
                    scores_temp = pd.DataFrame({'Model': [describe]})
                    for idx in temp.index:
                        scores_temp[idx] = temp.loc[idx]
                    scores = pd.concat([scores, scores_temp])
                    return scores
                print(f'Mean: {scores.iloc[:,1:].mean()}')
                print(f'Std: {scores.iloc[:, 1:].std()}')

                log_file = os.path.join(args.save_path, 'log.log')
                with open(log_file, 'a') as f:
                    print(f'Mean: {scores.iloc[:,1:].mean()}', file=f)
                    print(f'Std: {scores.iloc[:, 1:].std()}', file=f)
                scores = func_mean_std(scores.iloc[:,1:].mean(), scores,'mean')
                scores = func_mean_std(scores.iloc[:, 1:].std(), scores,'std')

                scores.to_csv(args.save_path+ '/' + 'scores.csv',index=False)
                a=0
