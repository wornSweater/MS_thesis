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
    def __init__(self, filepath: str, target: str, test_size=0.2, val_size = 0.1):

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
        elif self.target == 'straddle_return' or self.target == 'settlement_value':
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
        
        # Setup scheduler
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
        train_mlp_model(model, train_loader, val_loader, optimizer, criterion, scheduler, best_params['epochs'], device)
        self.model_ = model
        return self.model_
    
    def lstm(self, best_params):
        # Apply scaling
        self.model = 'lstm'
        
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_transform_tree_nn(best_params)
        self.X_train_scaled = X_train_scaled
        self.X_val_scaled = X_val_scaled
        self.X_test_scaled = X_test_scaled
        
        # Reconstruct FC layers
        fc_layers = []
        if best_params['use_fc_layers']:
            for i in range(best_params.get('num_fc_layers', 0)):
                fc_size = best_params.get(f'fc_layer_{i}_size')
                if fc_size:
                    fc_layers.append(fc_size)
        
        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMRegressor(
            input_dim=X_train_scaled.shape[1],
            hidden_dim=best_params['hidden_dim'],
            num_layers=best_params['num_lstm_layers'],
            dropout_rate=best_params['dropout_rate'],
            bidirectional=best_params['bidirectional'],
            batch_norm=best_params['batch_norm'],
            activation=best_params['activation'],
            fc_layers=fc_layers if fc_layers else None,
            lstm_dropout=best_params.get('lstm_dropout', 0.0)
        ).to(device)
        
        # Create datasets
        sequence_length = best_params['sequence_length']
        train_dataset = TimeSeriesLSTMDataset(X_train_scaled.to_numpy(), self.y_train.to_numpy(), sequence_length)
        val_dataset = TimeSeriesLSTMDataset(X_val_scaled.to_numpy(), self.y_val.to_numpy(), sequence_length)
        
        train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)
        
        # Setup optimizer
        optimizer_name = best_params['optimizer']
        if optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
        elif optimizer_name == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
        elif optimizer_name == "RMSprop":
            optimizer = optim.RMSprop(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])

        # Setup scheduler
        scheduler = None
        scheduler_name = best_params['scheduler']
        if scheduler_name == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        elif scheduler_name == "ExponentialLR":
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=best_params['gamma'])
        elif scheduler_name == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=best_params['step_size'], gamma=0.1)
        else:
            scheduler = None
        
        criterion = nn.MSELoss()
        
        # Train model
        train_lstm_model(model, train_loader, val_loader, optimizer, criterion, scheduler, best_params['epochs'], device)
        self.model_ = model
        return self.model_

    def gru(self, best_params):
        # Apply scaling
        self.model = 'gru'
        
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_transform_tree_nn(best_params)
        self.X_train_scaled = X_train_scaled
        self.X_val_scaled = X_val_scaled
        self.X_test_scaled = X_test_scaled
        
        # Reconstruct FC layers
        fc_layers = []
        if best_params['use_fc_layers']:
            for i in range(best_params.get('num_fc_layers', 0)):
                fc_size = best_params.get(f'fc_layer_{i}_size')
                if fc_size:
                    fc_layers.append(fc_size)
        
        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GRURegressor(
            input_dim=X_train_scaled.shape[1],
            hidden_dim=best_params['hidden_dim'],
            num_layers=best_params['num_gru_layers'],
            dropout_rate=best_params['dropout_rate'],
            bidirectional=best_params['bidirectional'],
            batch_norm=best_params['batch_norm'],
            activation=best_params['activation'],
            fc_layers=fc_layers if fc_layers else None,
            gru_dropout=best_params.get('gru_dropout', 0.0)
        ).to(device)
        
        # Create datasets
        sequence_length = best_params['sequence_length']
        train_dataset = TimeSeriesGRUDataset(X_train_scaled.to_numpy(), self.y_train.to_numpy(), sequence_length)
        val_dataset = TimeSeriesGRUDataset(X_val_scaled.to_numpy(), self.y_val.to_numpy(), sequence_length)
        
        train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)
        
        # Setup optimizer
        optimizer_name = best_params['optimizer']
        if optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
        elif optimizer_name == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
        elif optimizer_name == "RMSprop":
            optimizer = optim.RMSprop(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])

        # Setup scheduler
        scheduler = None
        scheduler_name = best_params['scheduler']
        if scheduler_name == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        elif scheduler_name == "ExponentialLR":
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=best_params['gamma'])
        elif scheduler_name == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=best_params['step_size'], gamma=0.1)
        else:
            scheduler = None
        
        criterion = nn.MSELoss()
        
        # Train model
        train_gru_model(model, train_loader, val_loader, optimizer, criterion, scheduler, best_params['epochs'], device)
        self.model_ = model
        return self.model_

# --------- Custom Dataset for MLP ----------
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

# --------- MLP Modle Constructor ----------
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
    

# --------- Train MLP Model ----------
def train_mlp_model(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs, device, early_stopping_patience=10):
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


# --------- Custom Dataset for Time Series LSTM ----------
class TimeSeriesLSTMDataset(Dataset):
    def __init__(self, X, y, sequence_length):
        self.sequence_length = sequence_length
        
        # Convert to numpy arrays first to ensure consistent handling
        if hasattr(X, 'values'):  # pandas DataFrame
            X = X.values
        if hasattr(y, 'values'):  # pandas Series
            y = y.values
            
        # Ensure y is 1D
        if y.ndim > 1:
            y = y.squeeze()
            
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
        # Create sequences for LSTM
        self.sequences_X, self.sequences_y = self._create_sequences()
    
    def _create_sequences(self):
        sequences_X = []
        sequences_y = []
        
        # Ensure we have enough data for at least one sequence
        if len(self.X) < self.sequence_length:
            return torch.empty(0, self.sequence_length, self.X.shape[1]), torch.empty(0)
        
        for i in range(len(self.X) - self.sequence_length + 1):
            seq_x = self.X[i:i + self.sequence_length]
            seq_y = self.y[i + self.sequence_length - 1]  # Predict the last value in sequence
            
            # Ensure seq_y is a scalar, not an array
            if hasattr(seq_y, 'item'):
                seq_y = seq_y.item()
            
            sequences_X.append(seq_x)
            sequences_y.append(seq_y)
        
        if len(sequences_X) == 0:
            return torch.empty(0, self.sequence_length, self.X.shape[1]), torch.empty(0)
        
        return torch.stack(sequences_X), torch.tensor(sequences_y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.sequences_X)
    
    def __getitem__(self, idx):
        return self.sequences_X[idx], self.sequences_y[idx]


# --------- LSTM Model Architecture ----------
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate=0.2, 
                 bidirectional=False, batch_norm=True, activation='relu', 
                 fc_layers=None, lstm_dropout=0.0):
        super(LSTMRegressor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout if num_layers > 1 else 0,  # LSTM dropout only works with >1 layer
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Activation function mapping
        activation_functions = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'mish': nn.Mish()
        }
        
        # Fully connected layers after LSTM
        fc_layers_list = []
        prev_dim = lstm_output_dim
        
        if fc_layers:
            for fc_dim in fc_layers:
                fc_layers_list.append(nn.Linear(prev_dim, fc_dim))
                
                if batch_norm:
                    fc_layers_list.append(nn.BatchNorm1d(fc_dim))
                
                fc_layers_list.append(activation_functions[activation])
                
                if dropout_rate > 0:
                    fc_layers_list.append(nn.Dropout(dropout_rate))
                
                prev_dim = fc_dim
        
        # Output layer
        fc_layers_list.append(nn.Linear(prev_dim, 1))
        
        self.fc_layers = nn.Sequential(*fc_layers_list)
        
        # Initialize weights
        self.apply(lambda m: self._init_weights(m, activation))
    
    def _init_weights(self, module, activation):
        """Initialize weights based on activation function"""
        if isinstance(module, nn.Linear):
            if activation in ['tanh', 'sigmoid']:
                nn.init.xavier_uniform_(module.weight)
            elif activation in ['relu', 'leaky_relu', 'elu']:
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            else:
                nn.init.xavier_uniform_(module.weight)
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last time step output
        # lstm_out shape: (batch_size, sequence_length, hidden_dim * num_directions)
        last_output = lstm_out[:, -1, :]  # Take last time step
        
        # Pass through fully connected layers
        output = self.fc_layers(last_output)
        
        return output.squeeze()


# --------- Training LSTM Function ----------
def train_lstm_model(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs, device, early_stopping_patience=10):
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in tqdm(range(epochs)):
        # --------- Training ----------
        train_loss = 0.0
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Ensure outputs and targets have compatible shapes
            if outputs.dim() > 1:
                outputs = outputs.squeeze()
            if batch_y.dim() > 1:
                batch_y = batch_y.squeeze()
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping (important for LSTMs)
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
                
                # Ensure outputs and targets have compatible shapes
                if outputs.dim() > 1:
                    outputs = outputs.squeeze()
                if batch_y.dim() > 1:
                    batch_y = batch_y.squeeze()
                
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


# --------- Custom Dataset for Time Series GRU ----------
class TimeSeriesGRUDataset(Dataset):
    def __init__(self, X, y, sequence_length):
        self.sequence_length = sequence_length
        
        # Convert to numpy arrays first to ensure consistent handling
        if hasattr(X, 'values'):  # pandas DataFrame
            X = X.values
        if hasattr(y, 'values'):  # pandas Series
            y = y.values
            
        # Ensure arrays are at least 2D and 1D respectively
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim > 1:
            y = y.squeeze()
        if y.ndim == 0:
            y = np.array([y])
            
        # Convert to tensors
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
        # Validate we have enough data
        if len(self.X) < self.sequence_length:
            print(f"Warning: Data length {len(self.X)} < sequence length {self.sequence_length}")
            
        # Create sequences for GRU
        self.sequences_X, self.sequences_y = self._create_sequences()
    
    def _create_sequences(self):
        sequences_X = []
        sequences_y = []
        
        # Ensure we have enough data for at least one sequence
        if len(self.X) < self.sequence_length:
            empty_shape = (0, self.sequence_length, self.X.shape[1] if self.X.dim() > 1 else 1)
            return torch.empty(empty_shape), torch.empty(0)
        
        for i in range(len(self.X) - self.sequence_length + 1):
            seq_x = self.X[i:i + self.sequence_length]
            seq_y = self.y[i + self.sequence_length - 1]  # Predict the last value in sequence
            
            # Ensure seq_y is a scalar
            if hasattr(seq_y, 'item'):
                seq_y = seq_y.item()
            elif torch.is_tensor(seq_y) and seq_y.numel() == 1:
                seq_y = seq_y.item()
            elif isinstance(seq_y, np.ndarray) and seq_y.size == 1:
                seq_y = float(seq_y.item())
            else:
                seq_y = float(seq_y)
            
            sequences_X.append(seq_x)
            sequences_y.append(seq_y)
        
        if len(sequences_X) == 0:
            empty_shape = (0, self.sequence_length, self.X.shape[1] if self.X.dim() > 1 else 1)
            return torch.empty(empty_shape), torch.empty(0)
        
        try:
            return torch.stack(sequences_X), torch.tensor(sequences_y, dtype=torch.float32)
        except Exception as e:
            print(f"Error stacking sequences: {e}")
            print(f"Sequence X shapes: {[seq.shape for seq in sequences_X[:3]]}")
            print(f"Sequence y types: {[type(seq) for seq in sequences_y[:3]]}")
            raise
    
    def __len__(self):
        return len(self.sequences_X)
    
    def __getitem__(self, idx):
        return self.sequences_X[idx], self.sequences_y[idx]
    

# --------- GRU Model Architecture ----------
class GRURegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate=0.2, 
                 bidirectional=False, batch_norm=True, activation='relu', 
                 fc_layers=None, gru_dropout=0.0):
        super(GRURegressor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # GRU layers - simpler than LSTM (no cell state)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=gru_dropout if num_layers > 1 else 0,  # GRU dropout only works with >1 layer
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate GRU output dimension
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Activation function mapping
        activation_functions = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'mish': nn.Mish()
        }
        
        # Fully connected layers after GRU
        fc_layers_list = []
        prev_dim = gru_output_dim
        
        if fc_layers and len(fc_layers) > 0:
            for fc_dim in fc_layers:
                fc_layers_list.append(nn.Linear(prev_dim, fc_dim))
                
                if batch_norm:
                    fc_layers_list.append(nn.BatchNorm1d(fc_dim))
                
                fc_layers_list.append(activation_functions[activation])
                
                if dropout_rate > 0:
                    fc_layers_list.append(nn.Dropout(dropout_rate))
                
                prev_dim = fc_dim
        
        # Output layer (no activation for regression)
        fc_layers_list.append(nn.Linear(prev_dim, 1))
        
        self.fc_layers = nn.Sequential(*fc_layers_list)
        
        # Initialize weights
        self.apply(lambda m: self._init_weights(m, activation))
    
    def _init_weights(self, module, activation):
        """Initialize weights based on activation function"""
        if isinstance(module, nn.Linear):
            if activation in ['tanh', 'sigmoid']:
                nn.init.xavier_uniform_(module.weight)
            elif activation in ['relu', 'leaky_relu', 'elu']:
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            else:
                nn.init.xavier_uniform_(module.weight)
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
        elif isinstance(module, nn.GRU):
            # GRU weight initialization
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        batch_size = x.size(0)
        
        # GRU forward pass (simpler than LSTM - only hidden state, no cell state)
        gru_out, hidden = self.gru(x)
        
        # Use the last time step output
        # gru_out shape: (batch_size, sequence_length, hidden_dim * num_directions)
        last_output = gru_out[:, -1, :]  # Take last time step
        
        # Pass through fully connected layers
        output = self.fc_layers(last_output)
        
        # Ensure output is properly shaped
        if output.dim() > 1:
            output = output.squeeze(-1)
        
        return output
    

# --------- Training GRU Function ----------
def train_gru_model(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs, device, early_stopping_patience=10):
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in tqdm(range(epochs)):
        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_y in train_loader:
            try:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                # Ensure proper shapes
                if batch_X.dim() != 3:
                    print(f"Warning: Unexpected batch_X shape: {batch_X.shape}")
                    continue
                    
                if batch_y.dim() > 1:
                    batch_y = batch_y.squeeze()
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                
                # Ensure outputs and targets have compatible shapes
                if outputs.dim() > 1:
                    outputs = outputs.squeeze()
                if batch_y.dim() > 1:
                    batch_y = batch_y.squeeze()
                
                # Skip if shapes are incompatible
                if outputs.shape != batch_y.shape:
                    print(f"Shape mismatch: outputs {outputs.shape}, targets {batch_y.shape}")
                    continue
                
                loss = criterion(outputs, batch_y)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print("NaN loss detected, skipping batch")
                    continue
                    
                loss.backward()
                
                # Gradient clipping (important for RNNs)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        
        if num_batches == 0:
            print("No valid training batches processed")
            break
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                try:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    if batch_X.dim() != 3:
                        continue
                        
                    if batch_y.dim() > 1:
                        batch_y = batch_y.squeeze()
                    
                    outputs = model(batch_X)
                    
                    # Ensure outputs and targets have compatible shapes
                    if outputs.dim() > 1:
                        outputs = outputs.squeeze()
                    if batch_y.dim() > 1:
                        batch_y = batch_y.squeeze()
                    
                    if outputs.shape != batch_y.shape:
                        continue
                    
                    loss = criterion(outputs, batch_y)
                    
                    if not torch.isnan(loss):
                        val_loss += loss.item()
                        val_batches += 1
                        
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        if val_batches == 0:
            print("No valid validation batches processed")
            break
            
        val_loss /= val_batches
        
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