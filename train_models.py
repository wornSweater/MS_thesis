# basics
import optuna
import json
import gc
import numpy as np
import pandas as pd
import joblib
import shap
from tqdm import tqdm

# tree based
import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# linear with regularization
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# data processing and performance metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

# features
from feature_settings import *

class Train_Models:
    def __init__(self, filepath: str, target: str, test_size=0.2, val_size = 0.05):

        self.filepath = filepath
        self.target = target
        self.test_size = test_size
        self.val_size = val_size
        self.scalers = {
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
            "robust": RobustScaler,
            "none": None
        }
        self.features = None
        self.data = None

        self.X = None
        self.y = None
        
        self.X_full_train = None
        self.X_train = None
        self.X_val = None
        self.X_test = None

        self.y_full_train = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

        self.X_train_scaled = None
        self.X_val_scaled = None
        self.X_test_scaled = None

        self.model_ = None
        self.model = 'Empty'
        self.which_data = None

        try:
            self.data = pd.read_csv(filepath)
        except Exception as e:
            print("===== Fail to read the data, please check the filename and file format =====")

        if self.target == 'option_return':
            self.features = FEATURES_WHOLE_SCALED
            self.which_data = 'whole'
        elif self.target == 'straddle_return':
            self.features = FEATURES_STRADDLE_SCALED
            self.which_data = 'straddle'
        else:
            raise ValueError("===== The target input is wrong, please change to the right one =====")
        
        self.X = self.data[self.data.columns[:-1]]
        self.y = self.data[self.target]
        assert self.data.columns[-1] == self.target, "===== Target column is wrong ====="
        self.X_full_train, self.X_test, self.y_full_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, shuffle=False)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_full_train, self.y_full_train, test_size=self.val_size, shuffle=False)
        
    def get_X_train_scaled(self):    
        return self.X_train_scaled
    
    def get_X_test_scaled(self):
        return self.X_test_scaled
    
    def get_y_full_train(self):
        return self.y_full_train
    
    def get_y_train(self):
            return self.y_train

    def get_y_test(self):
        return self.y_test

    def scale_transform_linear(self, params):
        if 'scaler' not in params.keys():
            raise ValueError("===== The scaler is not specified in params =====")
        
        if self.scalers[params['scaler']]:
            scaler_class = self.scalers[params['scaler']]
            ct = ColumnTransformer(
                transformers=[('scale', scaler_class(), self.features)],
                remainder='passthrough' 
            )
            X_full_train_scaled = pd.DataFrame(
                ct.fit_transform(self.X_full_train), 
                columns=self.X_full_train.columns, 
                index=self.X_full_train.index
            )
            X_test_scaled = pd.DataFrame(
                ct.transform(self.X_test), 
                columns=self.X_test.columns, 
                index=self.X_test.index
            )
        else:
            X_full_train_scaled = self.X_full_train
            X_test_scaled = self.X_test
        return X_full_train_scaled, X_test_scaled

    def scale_transform_tree_nn(self, params):
        if 'scaler' not in params.keys():
            raise ValueError("===== The scaler is not specified in params =====")
        
        if self.scalers[params['scaler']]:
            scaler_class = self.scalers[params['scaler']]
            ct = ColumnTransformer(
                transformers=[('scale', scaler_class(), self.features)],
                remainder='passthrough' 
            )
            X_train_scaled = pd.DataFrame(
                ct.fit_transform(self.X_train), 
                columns=self.X_train.columns, 
                index=self.X_train.index
            )
            X_val_scaled = pd.DataFrame(
                ct.transform(self.X_val), 
                columns=self.X_val.columns, 
                index=self.X_val.index
            )
            X_test_scaled = pd.DataFrame(
                ct.transform(self.X_test), 
                columns=self.X_test.columns, 
                index=self.X_test.index
            )
        else:
            X_train_scaled = self.X_train
            X_val_scaled = self.X_val
            X_test_scaled = self.X_test
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def metrics(self, y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape =  mean_absolute_percentage_error(y_true, y_pred)
        return r2, mse, mae, rmse, mape
        
    def linear(self):
        self.model = "linear"

        self.X_train_scaled = self.X_full_train
        self.X_test_scaled = self.X_test

        model = LinearRegression()
        model.fit(self.X_full_train, self.y_full_train)
        self.model_ = model
        return self.model_

    def lasso(self, params):
        self.model = "lasso"

        if 'scaler' not in params.keys():
            raise ValueError("===== The scaler is not specified in params =====")
        
        X_full_train_scaled, X_test_scaled = self.scale_transform_linear(params)
        self.X_train_scaled = X_full_train_scaled
        self.X_test_scaled = X_test_scaled

        del params['scaler']

        model = Lasso(**params)
        model.fit(self.X_train_scaled, self.y_full_train)
        self.model_ = model
        return self.model_
    
    def ridge(self, params):
        self.model = "ridge"

        if 'scaler' not in params.keys():
            raise ValueError("===== The scaler is not specified in params =====")
        
        X_full_train_scaled, X_test_scaled = self.scale_transform_linear(params)
        self.X_train_scaled = X_full_train_scaled
        self.X_test_scaled = X_test_scaled

        del params['scaler']

        model = Ridge(**params)
        model.fit(self.X_train_scaled, self.y_full_train)
        self.model_ = model
        return self.model_
    
    def elastic(self, params):
        self.model = "elastic"

        if 'scaler' not in params.keys():
            raise ValueError("===== The scaler is not specified in params =====")
        
        X_full_train_scaled, X_test_scaled = self.scale_transform_linear(params)
        self.X_train_scaled = X_full_train_scaled
        self.X_test_scaled = X_test_scaled

        del params['scaler']

        model = ElasticNet(**params)
        model.fit(self.X_train_scaled, self.y_full_train)
        self.model_ = model
        return self.model_
    
    def xgb(self, params):
        self.model = "xgb"

        if 'scaler' not in params.keys():
            raise ValueError("===== The scaler is not specified in params =====")
        
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_transform_tree_nn(params)
        self.X_train_scaled = X_train_scaled
        self.X_val_scaled = X_val_scaled
        self.X_test_scaled = X_test_scaled

        del params['scaler']

        model = XGBRegressor(**params)
        model.fit(
            self.X_train_scaled, self.y_train,
            eval_set=[(self.X_val_scaled, self.y_val)],
            verbose=False
        )
        self.model_ = model
        return self.model_
    
    def lgb(self, params):
        self.model = "lgb"

        if 'scaler' not in params.keys():
            raise ValueError("===== The scaler is not specified in params =====")
        
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_transform_tree_nn(params)
        self.X_train_scaled = X_train_scaled
        self.X_val_scaled = X_val_scaled
        self.X_test_scaled = X_test_scaled

        del params['scaler']

        model = LGBMRegressor(**params)
        model.fit(
            self.X_train_scaled, self.y_train,
            eval_set=[(self.X_val_scaled, self.y_val)],
            eval_metric='rmse'
        )
        self.model_ = model
        return self.model_
    
    def cat(self, params):
        self.model = "cat"

        if 'scaler' not in params.keys():
            raise ValueError("===== The scaler is not specified in params =====")
        
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_transform_tree_nn(params)
        self.X_train_scaled = X_train_scaled
        self.X_val_scaled = X_val_scaled
        self.X_test_scaled = X_test_scaled

        del params['scaler']

        model = CatBoostRegressor(**params)
        if self.which_data == 'whole':
            model.fit(self.X_train_scaled, self.y_train, cat_features=FEATURES_INDICATOR)
        else:
            model.fit(
                self.X_train_scaled, self.y_train,
                eval_set=(self.X_val_scaled, self.y_val),
                use_best_model=True
            )
        self.model_ = model
        return self.model_
    
    def mlp(self, best_params):
        self.model = 'mlp'
        
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_transform_tree_nn(best_params)
        self.X_train_scaled = X_train_scaled
        self.X_val_scaled = X_val_scaled
        self.X_test_scaled = X_test_scaled
        
        # Reconstruct architecture
        hidden_layers = [best_params['first_layer_size']]
        for i in range(1, best_params['num_layers']):
            layer_ratio = best_params.get(f'layer_{i}_ratio', 1.0)
            layer_size = max(16, int(hidden_layers[-1] * layer_ratio))
            hidden_layers.append(layer_size)
        
        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MLPRegressor(
            input_dim=X_train_scaled.shape[1],
            hidden_layers=hidden_layers,
            dropout_rate=best_params['dropout_rate'],
            batch_norm=best_params['batch_norm'],
            activation=best_params['activation']
        ).to(device)
        
        # Create dataloaders
        train_dataset = TimeSeriesDataset(X_train_scaled.to_numpy(), self.y_train.to_numpy())
        val_dataset = TimeSeriesDataset(X_val_scaled.to_numpy(), self.y_val.to_numpy())
        
        train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)
        
        # Setup optimizer and scheduler
        optimizer_name = best_params['optimizer']
        if optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
        elif optimizer_name == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
        elif optimizer_name == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'], momentum=0.9)
        elif optimizer_name == "RMSprop":
            optimizer = optim.RMSprop(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
        
        scheduler = None
        scheduler_name = best_params['scheduler']
        if scheduler_name == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        elif scheduler_name == "ExponentialLR":
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=best_params['gamma'])
        else:
            scheduler = None

        criterion = nn.MSELoss()
        
        # Train model
        train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, best_params['epochs'], device)
        self.model_ = model
        return self.model_


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate=0.2, batch_norm=True, activation='relu'):
        super(MLPRegressor, self).__init__()
        
        # Activation function mapping
        activation_functions = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),  # SiLU is the same as Swish
            'mish': nn.Mish()
        }
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for i, hidden_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Add chosen activation function
            layers.append(activation_functions[activation])
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer (no activation for regression)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights based on activation function
        self.apply(lambda m: self._init_weights(m, activation))
    
    def _init_weights(self, module, activation):
        """Initialize weights based on activation function"""
        if isinstance(module, nn.Linear):
            # Xavier/Glorot initialization for tanh and sigmoid
            if activation in ['tanh', 'sigmoid']:
                nn.init.xavier_uniform_(module.weight)
            # He initialization for ReLU and variants
            elif activation in ['relu', 'leaky_relu', 'elu']:
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            # Default Xavier for other activations
            else:
                nn.init.xavier_uniform_(module.weight)
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x).squeeze()
    

def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs, device, early_stopping_patience=20):
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in tqdm(range(epochs)):
        # Training
        train_loss = 0.0
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
        
        val_loss /= len(val_loader)
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break
    
    return best_val_loss