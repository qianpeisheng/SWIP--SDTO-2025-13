import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, ConstantKernel, RBF
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from termcolor import colored
from xgboost import XGBRegressor
import torch
from torch import optim
from torch.optim.swa_utils import AveragedModel, SWALR

# from torchcontrib.optim import SWA
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)  # Add this line for PyTorch reproducibility
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# setting the number of cross validations used in the Model part
nr_cv = 10
score_calc = 'neg_mean_squared_error'


def get_best_score(grid):
    best_score = np.sqrt(-grid.best_score_)
    print(best_score)
    print(grid.best_params_)
    print(grid.best_estimator_)
    return best_score


# # df_data has 846 columns
# df_data = pd.read_excel("data/SiC/training/SiC_Exp_Runs_Data_20250326.xlsx")

# edge_columns = ['Doping_10', 'Doping_12', 'Doping_13', 'Doping_16', 'Doping_18', 'Doping_19', 'Doping_22', 'Doping_24', 'Doping_25', 'Doping_28', 'Doping_3', 'Doping_30', 'Doping_33', 'Doping_34', 'Doping_36', 'Doping_39', 'Doping_4', 'Doping_40', 'Doping_42', 'Doping_45', 'Doping_46', 'Doping_48', 'Doping_6', 'Doping_7', 'Thickness_14', 'Thickness_21', 'Thickness_28', 'Thickness_35', 'Thickness_42', 'Thickness_49', 'Thickness_56', 'Thickness_63', 'Thickness_7', 'Thickness_70', 'Thickness_77', 'Thickness_84', 'X_10.1', 'X_12.1', 'X_13.1', 'X_14', 'X_16.1', 'X_18.1', 'X_19.1', 'X_21', 'X_22.1', 'X_24.1', 'X_25.1', 'X_28', 'X_28.1', 'X_3.1', 'X_30.1', 'X_33.1', 'X_34.1', 'X_35', 'X_36.1', 'X_39.1', 'X_4.1', 'X_40.1', 'X_42', 'X_42.1', 'X_45.1', 'X_46.1', 'X_48.1', 'X_49', 'X_56', 'X_6.1', 'X_63', 'X_7', 'X_7.1', 'X_70', 'X_77', 'X_84', 'Y_10.1', 'Y_12.1', 'Y_13.1', 'Y_14', 'Y_16.1', 'Y_18.1', 'Y_19.1', 'Y_21', 'Y_22.1', 'Y_24.1', 'Y_25.1', 'Y_28', 'Y_28.1', 'Y_3.1', 'Y_30.1', 'Y_33.1', 'Y_34.1', 'Y_35', 'Y_36.1', 'Y_39.1', 'Y_4.1', 'Y_40.1', 'Y_42', 'Y_42.1', 'Y_45.1', 'Y_46.1', 'Y_48.1', 'Y_49', 'Y_56', 'Y_6.1', 'Y_63', 'Y_7', 'Y_7.1', 'Y_70', 'Y_77', 'Y_84']

# # Remove edge
# df_data = df_data.drop(columns=edge_columns)
# # df_data Keep X and Y

# #input columns: inp_cols contains all step arrtibutes, has 480 attributes
# inp_cols = [col for col in df_data.columns if 'STEP' in col]
# #output columns: out_cols is the target, all thickness and dopping of remain points, here it is 122
# # out_cols = [col for col in df_data.columns if 'Thickness' in col or 'Doping' in col]
# out_cols = [col for col in df_data.columns if 'Thickness' in col ]

# # remaining_cols: the left columns after remain columns = 846 - 122 = 724 columns
# remaining_cols = df_data.columns.difference(out_cols)
# # XY_cols: remain points' coordinates columns： 244 columns, including 61 points (why 61 not 122 ?)
# XY_cols = remaining_cols.difference(inp_cols)
# print(XY_cols)

# # df_inps: the input including coordinates, 724 columns
# df_inps = df_data[remaining_cols]
# # df_inps: the input including coordinates, 61 points' thichness and
# df_outs = df_data[out_cols]

# # Identify columns with > 1 unique entries, which is useful attributes
# inp_unique_cols = [col for col in df_inps.columns if len(df_inps[col].unique()) > 1]

# # selecte the columns from Furen?
# inp_sel_cols = [col for col in inp_unique_cols if col not in ['STEP 2_FC041_CARBON_SX_Vent', 'STEP 2_FC060_TCS_SX_Vent', 'STEP 2_FC061_TCS_SX_Vent', 'STEP 2_EPR060_TCS_SX_Vent', 'STEP 3_TEMP_Start',
#                                                              'STEP 3_FC040_CARBON_SX_Vent', 'STEP 3_FC041_CARBON_SX_Vent', 'STEP 3_FC060_TCS_SX_Vent', 'STEP 3_FC061_TCS_SX_Vent', 'STEP 3_EPR060_TCS_SX_Vent',
#                                                              'STEP 3_FC970_DOPE2_Vent', 'STEP 3_FC971_DOPE2_INJ_CEN_Vent', 'STEP 3_FC974_DOPE2_INJ_LAT_Vent', 'STEP 3_FC975_DOPE2_DILUTE_Vent', 'STEP 3_EPC970_DOPE2_Vent',
#                                                              'STEP 4_TEMP_Start', 'STEP 5_TEMP_Stab', 'STEP 5_PRESSURE PID_Start', 'STEP 8_FC040_CARBON_SX_Start', 'STEP 8_FC040_CARBON_SX_Stab', 'STEP 8_FC041_CARBON_SX_Start',
#                                                               'STEP 8_FC041_CARBON_SX_Stab', 'STEP 8_FC060_TCS_SX_Start', 'STEP 8_FC060_TCS_SX_Stab', 'STEP 8_FC061_TCS_SX_Start', 'STEP 8_FC061_TCS_SX_Stab', 'STEP 9_FC014_H2_RUNLINE_Start',
#                                                               'STEP 9_FC016_H2_LAT_Start']]

# final_inp_columns = inp_sel_cols# + XY_cols.tolist()


# for i in range(len(df_data['STEP 3_TEMP_Ramp'])):
#     if isinstance(df_data.loc[i, 'STEP 3_TEMP_Ramp'], str):
#         time_vals = df_data.loc[i, 'STEP 3_TEMP_Ramp'].split(':')
#         tot_time_sec = float(time_vals[0])*60. + float(time_vals[1])
#     else:
#         tot_time_sec = 0.
#     df_data.loc[i, 'STEP 3_TEMP_Ramp'] = tot_time_sec
# df_data['STEP 3_TEMP_Ramp'] = df_data['STEP 3_TEMP_Ramp'].astype(float)
# print(df_data['STEP 3_TEMP_Ramp'])

# # the input and output dataframe
# df_sel_inp = df_data[final_inp_columns]
# df_sel_out = df_outs

# # # convert time to seconds
# # for i in range(len(df_sel_inp['STEP 3_TEMP_Ramp'])):
# #     if isinstance(df_sel_inp.loc[i, 'STEP 3_TEMP_Ramp'], str):
# #         time_vals = df_sel_inp.loc[i, 'STEP 3_TEMP_Ramp'].split(':')
# #         tot_time_sec = float(time_vals[0])*60. + float(time_vals[1])
# #     else:
# #         tot_time_sec = 0.
# #     df_sel_inp.loc[i, 'STEP 3_TEMP_Ramp'] = tot_time_sec
# # print(df_sel_inp['STEP 3_TEMP_Ramp'])

# all_data = pd.concat([df_sel_inp, df_sel_out], axis = 1)


# # # Create Dataset & Comparison of ML Approaches
# # X = df_sel_inp.values.astype(float)
# # Y = df_sel_out.values.astype(float)
# #
# # sc_in = StandardScaler()
# # sc_out = StandardScaler()
# # X_std = sc_in.fit_transform(X)
# # Y_std = sc_out.fit_transform(Y)

# print("data loaded")

# '''review data'''
# columns_lstm = [[] for _ in range(9)]

# for column in final_inp_columns:
#     for i in range(9):
#         if f"STEP {i+1}" in column:
#             columns_lstm[i].append(column)

# print("Valid Step in Precdtcion")
# for ix,column in enumerate(columns_lstm):
#     if len(column)!=0:
#         print(ix+1)

# '''data analysis done'''

# '''step 1,2,3,5,7'''
# valid_steps = ['STEP 1', 'STEP 2', 'STEP 3', 'STEP 5', 'STEP 7']
# X_inputs = [[] for _ in range(len(valid_steps))]
# columns_lstm_input = [[] for _ in range(len(valid_steps))]
# for column in final_inp_columns:
#     for ix, column_step in enumerate(valid_steps):
#         if column_step in column:
#             columns_lstm_input[ix].append(column)

# print("input step",valid_steps)
# input_shape=[]
# for ix,column in enumerate(columns_lstm_input):
#     print(f"steps {valid_steps[ix]}")
#     print(column)
#     X_inputs[ix] = df_sel_inp[column]
#     input_shape.append(len(column))

# print("input step",valid_steps)
# print("input shape: ",input_shape)


# # sc_in = StandardScaler()
# # sc_out = StandardScaler()
# # X_std = sc_in.fit_transform(X)
# # Y_std = sc_out.fit_transform(Y)
# '''normalize'''
# X_stds = [[] for _ in range(len(X_inputs))]
# X_StandardScaler_std = [[] for _ in range(len(X_inputs))]
# for ix,X_input in enumerate(X_inputs):
#     X_StandardScaler_std[ix] = StandardScaler()
#     X_stds[ix] = X_StandardScaler_std[ix].fit_transform(X_inputs[ix])

# Y = df_sel_out.values.astype(float)
# sc_out = StandardScaler()
# Y_std = sc_out.fit_transform(Y)
# # ... existing code ...


class MultiInputLSTM(nn.Module):
    def __init__(self, input_sizes, hidden_size, num_layers, output_size, dropout=0.2):
        super(MultiInputLSTM, self).__init__()

        # Create separate LSTM layers for each input

        self.fcs = nn.ModuleList([nn.Linear(size, hidden_size) for size in input_sizes])
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        # Final fully connected layers
        # self.fc1 = nn.Linear(total_lstm_output, hidden_size)
        self.fc1 = nn.Linear(hidden_size * len(input_sizes), hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x_list):
        fc_outputs = []
        for i, nn_fc in enumerate(self.fcs):
            out = nn_fc(x_list[i])
            fc_outputs.append(out)

        combined_nns = torch.cat(fc_outputs, dim=1)
        lstm_out, (hidden, cell) = self.lstm(combined_nns)
        combined_lstms = lstm_out.reshape(lstm_out.shape[0], -1)

        # Pass through fully connected layers
        x = F.relu(self.fc1(combined_lstms))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class torch_data_loader(torch.utils.data.Dataset):
    def __init__(self, X_inputs, Y):
        self.X = []
        for X_input in X_inputs:
            tensor = torch.FloatTensor(X_input)
            tensor = tensor.unsqueeze(1)  # Add sequence dimension
            self.X.append(tensor)
        self.Y = torch.tensor(Y, dtype=torch.float)
        self.num_inputs = len(self.X)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        # self.X_ = []
        # for i in range(len(self.X)):
        #     self.X_.append(self.X[i][idx])
        # return self.X_, self.Y[idx]
        return idx, [self.X[i][idx] for i in range(self.num_inputs)], self.Y[idx]


# # Initialize and train the model
# input_sizes = input_shape  # Your list of input sizes
# hidden_size = 16
# num_layers = 2
# output_size = Y.shape[1]
# batch_size  = 128
# epochs = 3000
# epoch_val = 20
# loss_function  = nn.MSELoss()
# # Prepare data
# train_dataset = torch_data_loader(X_stds, Y_std)
# train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# # Create and train model
# model = MultiInputLSTM(input_sizes, hidden_size, num_layers, output_size)

# kfold = KFold(n_splits=nr_cv, shuffle=False)

# model_path = "models/sic/thickness/lstm"
# os.makedirs(f"{model_path}", exist_ok=True)
# tb_writer = SummaryWriter(model_path)

# num_models = 1
# device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# test_loss = []

# pred_list_thickness_train = []
# true_list_thickness_train = []

# pred_list_thickness_test = []
# true_list_thickness_test = []

# pred_list_thickness_val = []
# true_list_thickness_val = []

# pred_list_doping = []
# true_list_doping = []
# test_ids_list = []

# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# for fold, (train_ids, test_ids) in enumerate(kfold.split(train_dataset)):

#     if fold < 9:
#         continue
#     # Print
#     print(f'FOLD {fold}')
#     print('--------------------------------')

#     # Calculate validation size to match test size
#     val_size = len(test_ids)
#     train_val_ids = train_ids
#     # # Randomly select validation indices from training set
#     # val_ids = np.random.choice(train_val_ids, size=val_size, replace=False)
#     # # Remove validation indices from training set
#     # train_ids = np.array([idx for idx in train_val_ids if idx not in val_ids])

#     # Calculate validation size to match test size
#     val_size = len(test_ids)
#     train_val_ids = train_ids
#     # # Randomly select validation indices from training set
#     # val_ids = np.random.choice(train_val_ids, size=val_size, replace=False)
#     # # Remove validation indices from training set
#     # train_ids = np.array([idx for idx in train_val_ids if idx not in val_ids])

#     # select validation indices from after testing set
#     val_ids = np.array([(idx + test_ids.shape[0]) % len(train_dataset) for idx in test_ids])
#     if fold == 9:
#         # to do 0,1,2,3 --> 0,1,2,3,4
#         val_ids = np.array([0, 1, 2, 3, 4])

#     # Create samplers for all three datasets
#     train_subset = torch.utils.data.Subset(train_dataset, train_ids)
#     val_subset = torch.utils.data.Subset(train_dataset, val_ids)
#     test_subset = torch.utils.data.Subset(train_dataset, test_ids)

#     # Define data loaders for training, validation and testing data
#     trainloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size,shuffle=True)
#     valloader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size,shuffle=True)
#     testloader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size,shuffle=False)

#     # # Sample elements randomly from a given list of ids, no replacement.
#     # train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
#     # test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
#     #
#     # # Define data loaders for training and testing data in this fold
#     # trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_subsampler)
#     # testloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=test_subsampler)

#     model_preds = []
#     model_idx = 0
#     r2score_thickness_old_train = 0
#     r2score_thickness_old_val = 0
#     r2score_thickness_old_test = 0
#     best_thickness_train = 0
#     best_thickness_val = 0
#     best_thickness_test = 0
#     loss_best_train = 100
#     loss_best_val = 100
#     loss_best_test = 100
#     models = [MultiInputLSTM(input_sizes, hidden_size, num_layers, output_size).to(device) for _ in range(num_models)]
#     for model in models:
#         print('Training Model - ', model_idx)
#         model.to(device)

#         # Initialize optimizer
#         # optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-10)  #, weight_decay=1e-10
#         optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-10)  # , weight_decay=1e-10
#         # optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=5e-6)  # , weight_decay=1e-10

#         for epoch in range(epochs):
#             model.train()

#             train_predictions = []
#             train_labels = []

#             for idx_train, x, y in trainloader:
#                 x_input = [x_tensor.to(device) for x_tensor in x]
#                 pred_train = model(x_input)
#                 loss_train = loss_function(pred_train, y.to(device))
#                 optimizer.zero_grad()
#                 loss_train.backward()
#                 optimizer.step()
#                 tb_writer.add_scalar(f'Folder-{fold+1}/train_loss', loss_train.item(), epoch)

#                 label_np_train = y.cpu().detach().numpy()
#                 pred_np_train = pred_train.cpu().detach().numpy()
#                 label_true_value_train = sc_out.inverse_transform(label_np_train)
#                 pred_true_value_train = sc_out.inverse_transform(pred_np_train)
#                 r2score_train = r2_score(label_true_value_train, pred_true_value_train)
#                 tb_writer.add_scalar(f'Folder-{fold + 1}/train_r2_score', r2score_train, epoch)

#                 train_predictions.append(pred_np_train)
#                 train_labels.append(label_np_train)

#             if (epoch % epoch_val == epoch_val - 1):
#                 print(colored("Fold {} - Epoch: {}  -  ".format(fold + 1, epoch + 1), "red"),
#                       colored(">>>>>>> Training MSELoss: {:.5f}   - ".format(loss_train.data), "green"), )

#                 model.eval()
#                 val_predictions = []
#                 val_labels = []
#                 total_val_loss = 0

#                 test_predictions = []
#                 test_labels = []
#                 total_test_loss = 0

#                 idx_tests = []
#                 with torch.no_grad():
#                     for idx_val, x_val, y_val in valloader:
#                         x_input_val = [x_tensor.to(device) for x_tensor in x_val]
#                         pred_val = model(x_input_val)
#                         loss_val = loss_function(pred_val, y_val.to(device))
#                         total_val_loss += loss_val.item()

#                         # Store predictions and labels
#                         val_predictions.append(pred_val.cpu().detach().numpy())
#                         val_labels.append(y_val.cpu().detach().numpy())

#                     for idx_test, x_test, y_test in testloader:
#                         x_input_test = [x_tensor.to(device) for x_tensor in x_test]
#                         pred_test = model(x_input_test)
#                         loss_test = loss_function(pred_test, y_test.to(device))
#                         total_test_loss += loss_test.item()

#                         # Store predictions and labels
#                         test_predictions.append(pred_test.cpu().detach().numpy())
#                         test_labels.append(y_test.cpu().detach().numpy())
#                         idx_tests.append(idx_test)


#                     # Combine all batches
#                     val_predictions = np.concatenate(val_predictions, axis=0)
#                     val_labels = np.concatenate(val_labels, axis=0)

#                     # Calculate average validation loss
#                     avg_val_loss = total_val_loss / len(valloader)
#                     tb_writer.add_scalar(f'Folder-{fold + 1}/val_loss', avg_val_loss, epoch)

#                     # Transform back to original scale
#                     label_true_value_val = sc_out.inverse_transform(val_labels)
#                     pred_true_value_val = sc_out.inverse_transform(val_predictions)

#                     # Calculate R2 score on full validation set
#                     r2score_val = r2_score(label_true_value_val, pred_true_value_val)
#                     tb_writer.add_scalar(f'Folder-{fold + 1}/val_r2_score', r2score_val, epoch)

#                     # Combine all batches
#                     test_predictions = np.concatenate(test_predictions, axis=0)
#                     test_labels = np.concatenate(test_labels, axis=0)
#                     idx_tests = np.concatenate(idx_tests, axis=0)

#                     # Calculate average test loss
#                     avg_test_loss = total_test_loss / len(testloader)
#                     tb_writer.add_scalar(f'Folder-{fold + 1}/test_loss', avg_test_loss, epoch)

#                     # Transform back to original scale
#                     label_true_value_test = sc_out.inverse_transform(test_labels)
#                     pred_true_value_test = sc_out.inverse_transform(test_predictions)
#                     # label_true_value_test[i] == Y[idx_test[i]]
#                     # label_true_value_test == Y[idx_test]

#                     # Calculate R2 score on full test set
#                     r2score_test = r2_score(label_true_value_test, pred_true_value_test)
#                     tb_writer.add_scalar(f'Folder-{fold + 1}/test_r2_score', r2score_test, epoch)

#                     train_predictions = np.concatenate(train_predictions, axis=0)
#                     train_labels = np.concatenate(train_labels, axis=0)

#                     label_true_value_train = sc_out.inverse_transform(train_labels)
#                     pred_true_value_train = sc_out.inverse_transform(train_predictions)

#                     if epoch == epoch_val - 1:
#                         best_thickness_val = pred_true_value_val
#                         r2score_thickness_old_val = r2score_val
#                         gt_thickness_val = label_true_value_val

#                         best_thickness_test = pred_true_value_test
#                         r2score_thickness_old_test = r2score_test
#                         gt_thickness_test = label_true_value_test

#                         best_thickness_train = pred_true_value_train
#                         gt_thickness_train = label_true_value_train

#                         torch.save(model.state_dict(), f"{model_path}/v7_73thick25_Fold_{fold}_{timestamp}.pth")

#                     # if avg_val_loss < loss_best_val:
#                     if r2score_thickness_old_val < r2score_val:
#                         print(f"save model in {model_path}/v7_73thick25_Fold_{fold}_{timestamp}.pth")
#                         print(f"get best loss from {loss_best_val} to {avg_val_loss}")
#                         print(f"get R2 {r2score_thickness_old_val} to {r2score_val}")

#                         loss_best_val = avg_val_loss
#                         r2score_thickness_old_val = r2score_val
#                         # torch.save(model.state_dict(), f"{model_path}/v7_73thick25_Fold_{fold}.pth")
#                         torch.save(model.state_dict(), f"{model_path}/v7_73thick25_Fold_{fold}_{timestamp}.pth")
#                         best_thickness_val = pred_true_value_val
#                         gt_thickness_val = label_true_value_val

#                         print(f"get best test loss from {loss_best_test} to {avg_test_loss}")
#                         print(f"get test R2 {r2score_thickness_old_test} to {r2score_test}")

#                         loss_best_test = avg_test_loss
#                         r2score_thickness_old_test = r2score_test
#                         best_thickness_test = pred_true_value_test
#                         gt_thickness_test = label_true_value_test
#                         r2score_thickness_old_test = r2score_test

#                         # loss_best_train = avg_train_loss
#                         best_thickness_train = pred_true_value_train
#                         gt_thickness_train = label_true_value_train
#                         torch.save(model.state_dict(), f"{model_path}/v7_73thick25_Fold_{fold}_{timestamp}.pth")

#                 # scheduler.step()
#         model_idx += 1
#         # model_idx += 1
#         # model.eval()
#         # torch.save(model.state_dict(), f"{model_path}/v7_73thick25doping_{fold}_{model_idx}.pth")
#         #
#         # with torch.no_grad():
#         #     for x_, y_ in testloader:     # how to confirm which one is test data in this fold?
#         #         pred = model(x_.to(device))
#         #         model_preds.append(pred.cpu().detach().numpy())
#         #
#         #         label_np = y_.cpu().detach().numpy()
#         #         pred_np =  pred.cpu().detach().numpy()
#         #         loss_test = mean_squared_error(label_np, pred_np)
#         #         r2score = r2_score(y_.cpu().detach().numpy(), pred.cpu().detach().numpy())
#     # Ensemble predictions using averaging
#     # ensemble_predictions = np.mean(model_preds, axis=0)
#     # loss_test = mean_squared_error(y_.cpu().detach().numpy(), ensemble_predictions)
#     print(f"get best R2 thickness {r2score_test} in fold-{fold}")
#     try:
#         pred_list_thickness_test.extend(best_thickness_test)
#         true_list_thickness_test.extend(gt_thickness_test)
#         test_ids_list.extend(idx_tests)

#         pred_list_thickness_val.extend(best_thickness_val)
#         true_list_thickness_val.extend(gt_thickness_val)

#         pred_list_thickness_train.extend(best_thickness_train)
#         true_list_thickness_train.extend(gt_thickness_train)
#     except:
#         print("bug")


#     # test_loss.extend([loss_test])

# pred_list_thickness_train = np.array(pred_list_thickness_train)
# true_list_thickness_train = np.array(true_list_thickness_train)

# pred_list_thickness_test = np.array(pred_list_thickness_test)
# true_list_thickness_test = np.array(true_list_thickness_test)
# test_ids_list = np.array(test_ids_list)

# pred_results = np.concatenate([test_ids_list.reshape(test_ids_list.shape[0],1),pred_list_thickness_test, true_list_thickness_test],axis=1)
# pred_results = pd.DataFrame(pred_results)

# # pred_results.to_csv(f"{model_path}/prediction_results.csv", index=False)
# pred_results.to_csv(f"{model_path}/prediction_results.csv", index=False)
# # pred_results = pred_results.sort_values(by=0)

# # a2 = pred_results2.iloc[:,-Y.shape[1]:]-df_sel_out.values.astype(float)

# # import pickle
# # with open(f"{model_path}/prediction_results.pkl", "wb") as f:
# #     pickle.dump([true_list_thickness_test,pred_list_thickness_test,true_list_doping,pred_list_doping], f)
# #
# r2_thickness_test = r2_score(true_list_thickness_test, pred_list_thickness_test)
# print(f"r2 thickness test: {r2_thickness_test}")

# mape_thickness_test = np.abs((true_list_thickness_test - pred_list_thickness_test) / true_list_thickness_test) * 100
# print(f"mape thickness test: {np.mean(mape_thickness_test)}")

# #val
# pred_list_thickness_val = np.array(pred_list_thickness_val)
# true_list_thickness_val = np.array(true_list_thickness_val)
# # val_ids_list = np.array(val_ids_list)
# r2_thickness_val = r2_score(true_list_thickness_val, pred_list_thickness_val)
# print(f"r2 thickness val: {r2_thickness_val}")

# mape_thickness_val = np.abs((true_list_thickness_val - pred_list_thickness_val) / true_list_thickness_val) * 100
# print(f"mape thickness val: {np.mean(mape_thickness_val)}")

# #train
# pred_list_thickness_train = np.array(pred_list_thickness_train)
# true_list_thickness_train = np.array(true_list_thickness_train)
# # train_ids_list = np.array(train_ids_list)
# r2_thickness_train = r2_score(true_list_thickness_train, pred_list_thickness_train)
# print(f"r2 thickness train: {r2_thickness_train}")

# mape_thickness_train = np.abs((true_list_thickness_train - pred_list_thickness_train) / true_list_thickness_train) * 100
# print(f"mape thickness train: {np.mean(mape_thickness_train)}")


# # #plot
# # from sklearn.metrics import r2_score
# # import matplotlib.pyplot as plt
# # from matplotlib.colors import LinearSegmentedColormap
# # from scipy.interpolate import griddata


# # # Extract columns starting from RM
# # start_col = 'X_0'
# # # df = df_data.loc[:, df_data.columns[df_data.columns.get_loc(start_col):]]
# # df = df_data.loc[:, df_data.columns[df_data.columns.get_loc(start_col):df_data.columns.get_loc('Thickness_83')]]

# # # Extract X, Y
# # x_cols = [col for col in df.columns if 'X' in col]
# # y_cols = [col for col in df.columns if 'Y' in col]

# # # Ensure we have valid X and Y coordinates
# # X = df_data[x_cols].values
# # Y = df_data[y_cols].values
# # avg_X = np.mean(X, axis=0)
# # avg_Y = np.mean(Y, axis=0)

# # colors = [
# #     (0.0, "blue"),  # Dark Blue at 0%
# #     (0.25, "cyan"),  # Light Blue at 25%
# #     (0.5, "lime"),  # Light Green at 50%
# #     (0.75, "yellow"),  # Yellow at 75%
# #     (1.0, "red"),  # Red at 100%
# # ]

# # reversed_colormap = LinearSegmentedColormap.from_list("reversed_colormap", colors)
# # cmap = reversed_colormap


# # norm_thres = 0.07

# # min_max_error = []
# # # plot_value = df_outs.values.astype(float)
# # plot_values = [true_list_thickness_test,pred_list_thickness_test,true_list_thickness_train,pred_list_thickness_train]
# # # plot_values = [pred_list_thickness_test]
# # save_names = ["gt_test","preds_test","gt_train","preds_train"]
# # for ix, plot_value in enumerate(plot_values):
# #     nrow, ncols = 6, 8
# #     fig, axes = plt.subplots(nrows=nrow, ncols=ncols, figsize=(24, 18))
# #     axes = axes.flatten()
# #     for i, ax in enumerate(axes):
# #         if i < plot_value.shape[0]:
# #             value = plot_value[i, :]
# #             max_norm = max(value)
# #             min_norm = min(value)
# #             # max_norm = 13.4
# #             # min_norm = 10.8
# #             if max_norm - min_norm < 0.3:
# #                 print(f"DOE{i + 1} : error {max_norm - min_norm}")
# #             min_max_error.append(max_norm - min_norm)
# #             ax.set_xlabel("DOE " + str(i + 1))
# #             norm = plt.Normalize(round(min_norm, 2), round(max_norm, 2))

# #             # Create a grid for the circle domain
# #             radius = max(X.flatten())
# #             grid_size = 1500  # Resolution of the grid
# #             x_grid = np.linspace(-radius, radius, grid_size)
# #             y_grid = np.linspace(-radius, radius, grid_size)
# #             x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

# #             # Circular mask
# #             mask = np.sqrt(x_mesh ** 2 + y_mesh ** 2) <= radius

# #             # Interpolate values onto the grid
# #             points = np.column_stack((avg_X.flatten(), avg_Y.flatten()))
# #             grid_thickness = griddata(points, value, (x_mesh, y_mesh), method='linear')

# #             # Apply the mask: Set values outside the circle to NaN
# #             grid_thickness[~mask] = np.nan

# #             # Plot heatmap
# #             ax.imshow(grid_thickness, extent=(-70, 70, -70, 70), origin='lower', cmap=cmap, norm=norm)
# #             # Scatter plot on top
# #             ax.scatter(avg_X, avg_Y, marker='x', alpha=0.3, c='red', s=10)
# #             ax.set_xticks([])
# #             ax.set_yticks([])
# #             ax.set_frame_on(False)
# #             # Add color bar for each subplot
# #             sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# #             sm.set_array([])
# #             cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
# #             cbar.set_label('Thickness')

# #     # Adjust layout to prevent overlap
# #     plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leaves space for suptitle
# #     # Save heatmap figure

# #     plt.show()
# #     plt.savefig(f"thickness_heatmap_{save_names[ix]}.png", dpi=300, bbox_inches='tight')
# #     plt.close()
