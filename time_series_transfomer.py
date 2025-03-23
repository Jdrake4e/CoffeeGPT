import file_readin_functions as frf
import utility_functions as uf
import numpy as np # type: ignore
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler # type: ignore
import pandas as pd # type: ignore
from itertools import product
import json
import os
import datetime
from pathlib import Path
import heapq
from model_analysis_test_set import plot_training_history

class TimeSeriesDataset:
    """
    A class for preparing and processing time series data for deep learning models.
    
    This class handles data preparation, scaling, and sequence creation for time series 
    forecasting tasks. It creates train, validation, and test splits and returns them 
    as PyTorch DataLoader objects.
    
    Parameters
    ----------
    sequence_size : int, default=365
        The length of each input sequence used for prediction.
    
    Attributes
    ----------
    sequence_size : int
        The length of each input sequence.
    scaler : sklearn.preprocessing.StandardScaler
        Scaler used to standardize features.
    x_test : torch.Tensor or None
        The test input sequences after processing.
    test_data : pandas.DataFrame or None
        The raw test data split.
    num_features : int or None
        The number of features in the processed dataset.
        
    Notes
    -----
    1. Data Processing Flow
    -----------------------
    1.1. Data Preparation
        - Handles splitting data into train, validation, and test sets
        - Performs standardization of all features
        - Creates sequence batches for time series modeling
    
    1.2. Data Validation
        - Performs checks for empty datasets after splitting
        - Validates data dimensions throughout the processing pipeline
    
    1.3. Sequence Creation
        - Converts raw data into overlapping sequences of specified length
        - Each sequence becomes an input sample, with the next value as target
    
    Examples
    --------
    >>> import pandas as pd
    >>> # Load time series data
    >>> data = pd.read_csv('stock_data.csv')
    >>> # Create dataset object with sequence length 30
    >>> dataset = TimeSeriesDataset(sequence_size=30)
    >>> # Prepare data with 80% training, 20% validation split
    >>> dataloaders = dataset.prepare_data(data, train_split=0.8, val_split=0.2)
    >>> # Access the prepared data loaders
    >>> train_loader = dataloaders['train']
    >>> val_loader = dataloaders['val']
    >>> test_loader = dataloaders['test']
    """
    def __init__(self, sequence_size=365):
        self.sequence_size = sequence_size
        self.scaler = StandardScaler()
        self.x_test = None
        self.test_data = None
        self.num_features = None
        
    def prepare_data(self, data, train_split=0.8, val_split=0.8):
        """
        Prepare time series data for model training and evaluation.
        
        This method handles data splitting, scaling, and sequence creation.
        It splits the data into train, validation, and test sets, scales the
        features, and creates overlapping sequences for time series prediction.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Input time series data with a 'Date' column and feature columns.
        train_split : float, default=0.8
            Proportion of data to use for training (0.0 to 1.0).
        val_split : float, default=0.8
            Proportion of training data to use for validation (0.0 to 1.0).
            
        Returns
        -------
        dict
            Dictionary containing DataLoader objects for 'train', 'val', and 'test' sets.
            
        Raises
        ------
        ValueError
            If any of the resulting datasets are empty after splitting.
        """
        
        #TODO find a nicer solution to this bugfix
        # DO NOT CHANGE THESE PRINTS FIXS A STUPID BUG WHERE test_data WAS BECOMING EMPTY
        # Still have no clue why this behavior is occuring
        print("Initial data shape:", data.shape)
    
        # Drop missing values and date column
        data = data.dropna()
        print("After dropna shape:", data.shape)
        
        # Calculate split indices
        train_idx = int(len(data) * train_split)
        print("Train index:", train_idx)
        print("Total data length:", len(data))
        
        self.test_data = data.iloc[train_idx:].copy()
        print("Test data shape:", self.test_data.shape)
        
        # First check if data is empty
        if len(data) == 0:
            raise ValueError("Input data is empty")
        
        # Calculate split indices
        train_idx = int(len(data) * train_split)
        if train_idx == 0:
            raise ValueError("Train split results in empty training set")
        
        # Drop missing values and date column
        data = data.dropna()
        self.test_data = data.iloc[train_idx:].copy()
        features = data.drop('Date', axis = 1)
        
        # Split data
        train_data = features.iloc[:train_idx]
        test_data = features.iloc[train_idx:]
        
        val_idx = int(len(train_data) * val_split)
        validation_data = train_data.iloc[val_idx:]
        train_data = train_data.iloc[:val_idx]
        
        if len(train_data) == 0:
            raise ValueError("Training data is empty after splitting")
        if len(validation_data) == 0:
            raise ValueError("Validation data is empty after splitting")
        if len(test_data) == 0:
            raise ValueError("Test data is empty after splitting")
        
        self.num_features = train_data.shape[1]  # Store number of features
        
        # Scale all features including target
        train_scaled = self._scale_data(train_data)
        val_scaled = self._scale_data(validation_data, fit=False)
        test_scaled = self._scale_data(test_data, fit=False)
        
        return self._create_data_loaders(train_scaled, val_scaled, test_scaled)
    
    def _scale_data(self, data, fit=True):
        """
        Scale data using StandardScaler.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Data to be scaled.
        fit : bool, default=True
            Whether to fit the scaler on this data or use previously fitted parameters.
            
        Returns
        -------
        numpy.ndarray
            Scaled data.
        """
        if fit:
            return self.scaler.fit_transform(data)
        return self.scaler.transform(data)
    
    def _create_sequences(self, data):
        """
        Create overlapping sequences for time series prediction.
        
        Takes input data and creates sequences of length `sequence_size`,
        with the target being the next value after each sequence.
        
        Parameters
        ----------
        data : numpy.ndarray
            Scaled input data.
            
        Returns
        -------
        tuple of torch.Tensor
            Input sequences and corresponding target values. 
            Index 0 will have the features and index 1 will have the target
        """
        num_features = data.shape[1]
        x, y = [], []
        
        # Create sequences for each feature
        for i in range(len(data) - self.sequence_size):
            # Extract sequence for all features
            sequence = data[i:(i + self.sequence_size)]
            target = data[i + self.sequence_size, num_features-1]  # Assuming Price is the first column
            
            # Reshape sequence to have features as the offset
            # Original shape: (sequence_size, num_features)
            # New shape: (sequence_size * num_features)
            sequence = sequence.flatten()
            
            x.append(sequence)
            y.append(target)
        
        # Reshape x to (num_sequences, sequence_size, num_features)
        x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, self.sequence_size, num_features)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
        return x_tensor, y_tensor
    
    def _create_data_loaders(self, train_data, val_data, test_data, batch_size=32):
        """
        Create PyTorch DataLoader objects from sequence data.
        
        Parameters
        ----------
        train_data : numpy.ndarray
            Scaled training data.
        val_data : numpy.ndarray
            Scaled validation data.
        test_data : numpy.ndarray
            Scaled test data.
        batch_size : int, default=32
            Batch size for the data loaders.
            
        Returns
        -------
        dict
            Dictionary containing DataLoader objects for 'train', 'val', and 'test' sets.
        """
        x_train, y_train = self._create_sequences(train_data)
        x_val, y_val = self._create_sequences(val_data)
        x_test, y_test = self._create_sequences(test_data)
        
        self.x_test = x_test
        
        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)
        test_dataset = TensorDataset(x_test, y_test)
        
        return {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            'val': DataLoader(val_dataset, batch_size=batch_size),
            'test': DataLoader(test_dataset, batch_size=batch_size)
        }

class PositionalEncoding(nn.Module):
    """
    Implements positional encoding for Transformer models.
    
    This module adds positional information to the input embeddings for sequence models,
    allowing the model to leverage sequence order information despite the parallelized
    nature of Transformer architectures.
    
    Parameters
    ----------
    d_model : int
        Dimension of the model's embeddings.
    dropout : float, default=0.1
        Dropout rate to apply after adding positional encodings.
    max_len : int, default=5000
        Maximum sequence length the model can handle.
        
    Notes
    -----
    1. Implementation Details
    -----------------------
    1.1. Sinusoidal Encoding
        - Uses sine and cosine functions of different frequencies for position encoding
        - Position enc(pos, 2i) = sin(pos/10000^(2i/d_model))
        - Position enc(pos, 2i+1) = cos(pos/10000^(2i/d_model))
    
    1.2. Benefits
        - Allows model to generalize to sequence lengths not seen during training
        - Provides consistent relative position information
        - Enables the model to capture sequential patterns and dependencies
    
    Examples
    --------
    >>> import torch
    >>> # Create positional encoding for embedding dimension 512
    >>> pos_encoder = PositionalEncoding(d_model=512, dropout=0.1, max_len=1000)
    >>> # Apply to an input tensor of shape [sequence_length, batch_size, embedding_dim]
    >>> input_tensor = torch.zeros(100, 32, 512)  # [seq_len, batch, d_model]
    >>> output = pos_encoder(input_tensor)
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and store
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input tensor.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [seq_len, batch_size, d_model]
            
        Returns
        -------
        torch.Tensor
            Tensor with added positional encodings, same shape as input
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ModelTuningLogger:
    """
    Logger for tracking model configurations and performance metrics during tuning.
    
    This class handles saving and loading experiment results, including model
    configurations, training history, and validation performance.
    
    Parameters
    ----------
    log_dir : str, default='logs'
        Directory to save experiment logs.
    experiment_name : str, optional
        Name of the experiment. If None, a timestamp will be used.
        
    Attributes
    ----------
    log_dir : Path
        Path to the log directory.
    experiment_name : str
        Name of the current experiment.
    results : dict
        Dictionary to store experiment results.
    """
    
    def __init__(self, log_dir='logs', experiment_name=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        if experiment_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"experiment_{timestamp}"
        else:
            self.experiment_name = experiment_name
            
        self.results = {}
        
    def log_experiment(self, model_config, training_history, val_metrics=None, test_metrics=None):
        """
        Log experiment details and results.
        
        Parameters
        ----------
        model_config : dict
            Dictionary containing model configuration parameters.
        training_history : dict
            Dictionary containing training history (losses, epochs, etc.).
        val_metrics : dict, optional
            Dictionary containing validation metrics.
        test_metrics : dict, optional
            Dictionary containing test metrics.
            
        Returns
        -------
        str
            Path to the saved JSON file.
        """
        # Generate a unique ID for this configuration
        config_id = f"config_{len(self.results) + 1}"
        
        # Create experiment record
        experiment = {
            "model_config": model_config,
            "training_history": training_history,
        }
        
        if val_metrics is not None:
            experiment["validation_metrics"] = val_metrics
            
        if test_metrics is not None:
            experiment["test_metrics"] = test_metrics
        
        # Add to results dictionary
        self.results[config_id] = experiment
        
        # Save to file
        return self.save_results()
    
    def save_results(self):
        """
        Save all experiment results to a JSON file.
        
        Returns
        -------
        str
            Path to the saved JSON file.
        """
        file_path = self.log_dir / f"{self.experiment_name}.json"
        
        try:
            # Convert results to JSON serializable format
            serializable_results = uf.convert_to_serializable(self.results)
            
            with open(file_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
                
        except TypeError as e:
            print(f"Warning: Could not serialize results directly: {e}")
            
        return str(file_path)
    
    def load_results(self, file_path=None):
        """
        Load experiment results from a JSON file.
        
        Parameters
        ----------
        file_path : str, optional
            Path to the JSON file. If None, uses the current experiment name.
            
        Returns
        -------
        dict
            Dictionary containing loaded experiment results.
        """
        if file_path is None:
            file_path = self.log_dir / f"{self.experiment_name}.json"
        else:
            file_path = Path(file_path)
            
        if file_path.exists():
            with open(file_path, 'r') as f:
                self.results = json.load(f)
        else:
            print(f"No results file found at {file_path}")
            
        return self.results
    
    def get_best_config(self, metric='val_loss', mode='min'):
        """
        Get the best configuration based on a specific metric.
        
        Parameters
        ----------
        metric : str, default='val_loss'
            Metric to use for comparison.
        mode : str, default='min'
            'min' for metrics where lower is better (e.g., loss),
            'max' for metrics where higher is better (e.g., accuracy).
            
        Returns
        -------
        tuple
            (config_id, config_dict) of the best configuration.
        """
        if not self.results:
            print("No results available")
            return None, None
            
        best_score = float('inf') if mode == 'min' else float('-inf')
        best_config_id = None
        
        for config_id, experiment in self.results.items():
            # Check if the metric exists in validation metrics
            if 'validation_metrics' in experiment and metric in experiment['validation_metrics']:
                score = experiment['validation_metrics'][metric]
            # Or check if it's in training history
            elif 'training_history' in experiment and metric in experiment['training_history']:
                # If it's a list, take the best value
                if isinstance(experiment['training_history'][metric], list):
                    if mode == 'min':
                        score = min(experiment['training_history'][metric])
                    else:
                        score = max(experiment['training_history'][metric])
                else:
                    score = experiment['training_history'][metric]
            else:
                continue
                
            if (mode == 'min' and score < best_score) or (mode == 'max' and score > best_score):
                best_score = score
                best_config_id = config_id
                
        if best_config_id is None:
            print(f"No configuration found with metric {metric}")
            return None, None
            
        return best_config_id, self.results[best_config_id]

#TODO Verify documentation again for this section
class TransformerModel(nn.Module):
    """
    Transformer-based model for time series forecasting.
    
    This model uses a Transformer encoder architecture to process multivariate
    time series data and predict future values.
    
    Parameters
    ----------
    input_dim : int
        Number of input features.
    d_model : int, default=512
        Dimension of the model embeddings and transformer layers.
    nhead : int, default=8
        Number of attention heads in transformer layers.
    num_layers : int, default=2
        Number of transformer encoder layers.
    dropout : float, default=0.1
        Dropout rate used throughout the model.
    decoder_config : dict, optional
        Configuration for the decoder network. If None, a default decoder is used.
        Expected keys:
        - hidden_layers: list of tuples (size, activation)
        - use_batch_norm: bool
        - dropout: float
        
    Notes
    -----
    1. Architecture Components
    ------------------------
    1.1. Input Encoding
        - Linear projection from input features to model dimension
        - Positional encoding to preserve sequence order information
    
    1.2. Transformer Encoder
        - Multi-head self-attention mechanism
        - Feedforward neural networks
        - Layer normalization and residual connections
    
    1.3. Output Decoder
        - Multi-layer feedforward neural network
        - Progressively reduces dimensions to final output
    
    2. Prediction Process
    -------------------
    2.1. Feature Encoding
        - Each timestep's features are encoded to the model dimension
    
    2.2. Sequence Processing
        - Transformer layers process the entire sequence in parallel
        - Self-attention captures dependencies between different timesteps
    
    2.3. Prediction Generation
        - The final timestep representation is used for prediction
        - Decoder network produces the final output value
    
    Examples
    --------
    >>> import torch
    >>> # Create a transformer model for time series with 5 features
    >>> model = TransformerModel(input_dim=5, d_model=256, nhead=4, num_layers=2)
    >>> # Process a batch of 32 sequences, each with 100 timesteps and 5 features
    >>> batch = torch.randn(32, 100, 5)  # [batch_size, seq_length, features]
    >>> output = model(batch)  # Shape: [batch_size, 1]
    """
    def __init__(self, input_dim, d_model=512, nhead=8, num_layers=2, dropout=0.1, decoder_config=None):
        super().__init__()
        
        # Modified encoder to handle multivariate input
        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True  # Important for handling batch-first input
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Create decoder based on configuration
        if decoder_config is None:
            # Default decoder configuration
            self.decoder = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.RReLU(lower=0.1, upper=0.3),
                nn.Linear(d_model // 2, 1)  # Fixed dimension calculation
            )
        else:
            # Custom decoder based on provided configuration
            layers = []
            input_size = d_model
            
            for i, (hidden_size, activation) in enumerate(decoder_config["hidden_layers"]):
                layers.append(nn.Linear(input_size, hidden_size))
                
                if activation is not None:  # Only add activation if specified
                    if activation == "relu":
                        layers.append(nn.ReLU())
                    elif activation == "leaky_relu":
                        layers.append(nn.LeakyReLU())
                    elif activation == "gelu":
                        layers.append(nn.GELU())
                    elif activation == "tanh":
                        layers.append(nn.Tanh())
                    elif activation == "silu":
                        layers.append(nn.SiLU())
                    elif activation == "elu":
                        layers.append(nn.ELU())
                    elif activation == "prelu":
                        layers.append(nn.PReLU())
                    elif activation == "linear":
                        # Use Identity for linear activation to make it explicit
                        layers.append(nn.Identity())
                    elif activation == "rrelu":
                        layers.append(nn.RReLU(lower=0.1, upper=0.3))
                    elif activation == "celu":
                        layers.append(nn.CELU(alpha=1.0))
                    elif activation == "softplus":
                        layers.append(nn.Softplus())
                
                if decoder_config.get("use_batch_norm", False) and i < len(decoder_config["hidden_layers"]) - 1:
                    layers.append(nn.BatchNorm1d(hidden_size))
                
                if decoder_config.get("dropout", 0) > 0 and i < len(decoder_config["hidden_layers"]) - 1:
                    layers.append(nn.Dropout(decoder_config["dropout"]))
                
                input_size = hidden_size
            
            # No need for output layer since it's included in hidden_layers
            self.decoder = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass through the transformer model.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, seq_length, num_features]
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch_size, 1] containing predictions
        """
        # Store the input sequence for use in loss calculation
        self.last_input_sequence = x.detach().clone()
        
        # Linear encoding of each timestep's features
        x = self.encoder(x)  # [batch_size, seq_length, d_model]
        
        # Apply positional encoding
        x = x.transpose(0, 1)  # [seq_length, batch_size, d_model]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # [batch_size, seq_length, d_model]
        
        # Pass through transformer
        x = self.transformer_encoder(x)  # [batch_size, seq_length, d_model]
        
        # Take the last sequence element and decode
        x = x[:, -1, :]  # [batch_size, d_model]
        x = self.decoder(x)  # [batch_size, 1]
        
        return x

    def generate_forecast(self, initial_sequence, num_steps, temperature=0.1, num_samples=100):
        """
        Generate multi-step forecasts with uncertainty quantification.
        
        Parameters
        ----------
        initial_sequence : torch.Tensor
            Initial sequence of shape [1, sequence_length, num_features]
        num_steps : int
            Number of steps to forecast into the future
        temperature : float, default=0.1
            Base temperature for uncertainty. Higher values = more uncertainty
        num_samples : int, default=100
            Number of Monte Carlo samples to generate
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'mean': Mean predictions for each future step
            - 'std': Standard deviation of predictions
            - 'samples': All Monte Carlo samples if num_samples > 1
        """
        self.eval()  # Set to evaluation mode
        device = next(self.parameters()).device
        
        # Ensure initial sequence is on correct device
        initial_sequence = initial_sequence.to(device)
        
        # Initialize storage for all samples
        all_samples = torch.zeros((num_samples, num_steps), device=device)
        
        with torch.no_grad():
            for sample in range(num_samples):
                current_sequence = initial_sequence.clone()
                
                for step in range(num_steps):
                    # Generate base prediction
                    prediction = self.forward(current_sequence)
                    
                    # Add noise scaled by step number (increasing uncertainty)
                    if num_samples > 1:  # Only add noise if we're doing Monte Carlo
                        step_temp = temperature * (1.0 + step * 0.1)  # Increase temperature with horizon
                        noise = torch.randn_like(prediction) * torch.sqrt(torch.tensor(step_temp))
                        prediction = prediction + noise
                    
                    # Store prediction
                    all_samples[sample, step] = prediction.squeeze()
                    
                    # Update sequence for next prediction
                    # Remove oldest timestep and add our prediction
                    current_sequence = torch.cat([
                        current_sequence[:, 1:, :],
                        torch.cat([
                            current_sequence[:, -1:, :-1],  # Keep other features unchanged
                            prediction.view(1, 1, 1)  # Add our new prediction
                        ], dim=2)
                    ], dim=1)
        
        # Calculate statistics
        mean_predictions = torch.mean(all_samples, dim=0)
        std_predictions = torch.std(all_samples, dim=0)
        
        return {
            'mean': mean_predictions.cpu().numpy(),
            'std': std_predictions.cpu().numpy(),
            'samples': all_samples.cpu().numpy() if num_samples > 1 else None
        }

#TODO Save json of all configurations and performance
#TODO Bring over other tuning grid from depricated version
def create_decoder_grid(d_model=512):
    """
    Create a grid of decoder configurations for hyperparameter tuning.

    This function generates a comprehensive set of decoder architectures by varying
    layer widths, activation functions, batch normalization, and dropout rates.
    The configurations are designed specifically for time series forecasting.

    Parameters
    ----------
    d_model : int, default=512
        Dimension of the transformer model's embeddings. This value is used to scale 
        the decoder layer widths proportionally.

    Returns
    -------
    list of dict
        List of decoder configurations. Each dictionary contains:
        
        - hidden_layers : list of tuples (size, activation)
            Layer architecture with sizes and activation types
        - use_batch_norm : bool
            Whether to use batch normalization between layers
        - dropout : float
            Dropout rate to apply between hidden layers
    """
    # Define the search space
    hidden_layer_widths = [
        [d_model // 2, d_model // 4],  # Default progressive reduction
        [d_model // 2, d_model // 4, d_model // 8],  # Deeper reduction
        #[d_model // 2],  # Single hidden layer
        #[d_model // 3, d_model // 9],  # Steeper reduction
        #[d_model // 2, d_model // 2, d_model // 4],  # Wide middle layer
        #[d_model // 2, d_model // 8, d_model // 4],  # Bottleneck architecture
        #[d_model, d_model // 2, d_model // 4],  # Wide initial layer
    ]
    
    # Optimized activation functions for time series
    activations = [
        #"gelu",      # Smooth gradient, works well with transformers
        #"elu",       # Good for time series, handles negative values well
        #"prelu",     # Learnable negative slope
        #"silu",      # Smooth, non-monotonic
        #"leaky_relu", # Simple but effective negative slope
        #"linear",    # Linear activation, good baseline for time series
        "rrelu"#,     # Randomized ReLU, good for noisy data
        #"celu",      # Continuously differentiable ELU
        #"softplus"   # Smooth approximation of ReLU
    ]
    
    use_batch_norm = [False] # [False, True] 
    dropout_rates = [0.2]
    
    # Generate all possible decoder configurations
    decoder_configs = []
    
    for width_config in hidden_layer_widths:
        # Try all combinations of the same activation through the network
        for act in activations:
            for batch_norm in use_batch_norm:
                for dropout in dropout_rates:
                    # Add hidden layers with activation
                    hidden_layers = [(size, act) for size in width_config]
                    # Add final output layer with no activation
                    hidden_layers.append((1, None))
                    
                    config = {
                        "hidden_layers": hidden_layers,
                        "use_batch_norm": batch_norm,
                        "dropout": dropout
                    }
                    
                    decoder_configs.append(config)
        
        #TODO Try all mixed activation functions, for now only first two architectures
        if width_config in [hidden_layer_widths[0], hidden_layer_widths[1]]:
            act_combinations = list(product(activations, repeat=len(width_config)))
            
            for acts in act_combinations:
                # Add hidden layers with mixed activations
                hidden_layers = [(width_config[i], act) for i, act in enumerate(acts)]
                # Add final output layer with no activation
                hidden_layers.append((1, None))
                
                config = {
                    "hidden_layers": hidden_layers,
                    "use_batch_norm": False,  # Fixed for mixed activations
                    "dropout": 0.2  # Fixed middle dropout for mixed activations
                }
                
                decoder_configs.append(config)
    
    return decoder_configs

#TODO Update Documentation
class ModelTrainer:
    """
    A comprehensive trainer class for time series forecasting models with PyTorch.
    
    This class manages the complete training lifecycle of deep learning models, including:
    - Model training with customizable loss functions
    - Validation and early stopping
    - Learning rate scheduling
    - Model evaluation
    - Forecasting with uncertainty quantification
    - Experiment logging and model checkpointing
    
    The trainer supports various loss functions optimized for time series forecasting:
    - MSE (default)
    - Custom weighted loss
    - Asymptote prevention loss
    - Adaptive forecast loss
    
    Features:
    - Automatic early stopping based on validation loss
    - Learning rate scheduling with ReduceLROnPlateau
    - Monte Carlo sampling for uncertainty quantification in forecasts
    - Experiment tracking and logging
    - Model checkpointing and best model saving
    - Comprehensive training history tracking
    
    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to train (typically a TransformerModel instance).
    device : torch.device
        The device to use for training (CPU or CUDA).
    learning_rate : float, default=1e-4
        Initial learning rate for the AdamW optimizer.
    model_config : dict, optional
        Configuration dictionary for model architecture and parameters.
        Used for experiment logging and reproducibility.
    logger : ModelTuningLogger, optional
        Logger instance for tracking experiments and saving results.
        If provided, will log training metrics, validation results,
        and model configurations.
    loss_config : dict, optional
        Configuration for the loss function. Should contain:
        - 'type': str, one of ['mse', 'custom', 'asymptote_prevention', 'adaptive_forecast']
        - Additional parameters specific to each loss type:
            - custom: {'alpha', 'beta', 'gamma'}
            - asymptote_prevention: {'historical_range', 'max_daily_change'}
            - adaptive_forecast: {'historical_range', 'forecast_steps', 'init_max_change',
                                'init_range_weight', 'init_change_weight'}
    
    Attributes
    ----------
    validation_loss_history : list
        History of validation losses during training.
    training_loss_history : list
        History of training losses during training.
    epoch_history : list
        List of completed epochs.
    lr_history : list
        History of learning rates during training.
    optimizer : torch.optim.AdamW
        The optimizer instance with weight decay.
    scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau
        Learning rate scheduler that reduces LR on plateau.
    criterion : torch.nn.Module
        The loss function based on loss_config.
    
    Methods
    -------
    train(data_loaders, epochs=1000, patience=10, model_dir='models', experiment_name=None)
        Main training loop with early stopping and checkpointing.
    train_epoch(train_loader)
        Trains the model for one epoch.
    validate(val_loader)
        Validates the model on validation data.
    evaluate(test_loader)
        Evaluates the model on test data.
    forecast(initial_sequence, num_steps, temperature=0.1, num_samples=100)
        Generates probabilistic forecasts with uncertainty quantification.
    """
    def __init__(self, model, device, learning_rate=1e-4, model_config=None, logger=None, loss_config=None):
        self.model = model
        self.device = device
        # Store model configuration, empty dict if no model_config provided
        self.model_config = model_config or {}
        # Initialize logger if provided
        self.logger = logger
        
        # Default loss configuration
        default_loss_config = {
            'type': 'mse'
        }
        self.loss_config = loss_config or default_loss_config
        
        # Initialize loss function and move to correct device
        self.criterion = self._initialize_loss_function().to(device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Initialize scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=10,
            verbose=False
        )
        
        # Initialize history trackers
        self.validation_loss_history = []
        self.training_loss_history = []
        self.epoch_history = []
        self.lr_history = []
        
    def _initialize_loss_function(self):
        """
        Initialize the loss function based on configuration.
        
        Returns
        -------
        torch.nn.Module
            Initialized loss function.
        
        Raises
        ------
        ValueError
            If loss type is not recognized or required parameters are missing.
        """
        loss_type = self.loss_config.get('type', 'mse')
        
        if loss_type == 'mse':
            return nn.MSELoss()
        
        elif loss_type == 'directional':
            return DirectionalLoss(
                direction_weight=self.loss_config.get('direction_weight', 0.3),
                mse_scale=self.loss_config.get('mse_scale', 10.0)
            )
        
        elif loss_type == 'custom':
            return CustomLoss(
                alpha=self.loss_config.get('alpha', 0.4),
                beta=self.loss_config.get('beta', 0.3),
                gamma=self.loss_config.get('gamma', 0.3)
            )
        
        elif loss_type == 'asymptote_prevention':
            if 'historical_range' not in self.loss_config:
                raise ValueError("historical_range required for asymptote_prevention loss")
            return AsymptotePreventionLoss(
                historical_range=self.loss_config['historical_range'],
                max_daily_change=self.loss_config.get('max_daily_change', 0.1)
            )
        
        elif loss_type == 'forecast_based':
            return ForecastBasedLoss(
                model=self.model,
                forecast_steps=self.loss_config.get('forecast_steps', 5),
                forecast_weight=self.loss_config.get('forecast_weight', 0.4),
                temperature=self.loss_config.get('temperature', 0.1),
                num_samples=self.loss_config.get('num_samples', 10)
            )
        
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    @staticmethod
    def get_available_loss_functions():
        """
        Get information about available loss functions and their parameters.
        
        Returns
        -------
        dict
            Dictionary containing loss function descriptions and required parameters.
        """
        return {
            'mse': {
                'description': 'Standard Mean Squared Error loss',
                'required_params': [],
                'optional_params': []
            },
            'directional': {
                'description': 'Combined MSE and cosine similarity loss for directional accuracy',
                'required_params': [],
                'optional_params': ['direction_weight', 'mse_scale']
            },
            'custom': {
                'description': 'Custom loss combining trend, volatility, and momentum',
                'required_params': [],
                'optional_params': ['alpha', 'beta', 'gamma']
            },
            'asymptote_prevention': {
                'description': 'Loss function preventing convergence to asymptotes',
                'required_params': ['historical_range'],
                'optional_params': ['max_daily_change']
            },
            'forecast_based': {
                'description': 'Loss function that uses model-generated forecasts to evaluate prediction quality',
                'required_params': [],
                'optional_params': [
                    'forecast_steps',     # Number of steps to forecast
                    'forecast_weight',    # Weight of forecast loss vs immediate loss
                    'temperature',        # Temperature for uncertainty
                    'num_samples'         # Number of Monte Carlo samples
                ]
            }
        }

    def train_epoch(self, train_loader):
        """
        Train the model for one epoch.
        
        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            DataLoader containing training data.
            
        Returns
        -------
        float
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0
        for batch in train_loader:
            x_batch, y_batch = [b.to(self.device) for b in batch]
            self.optimizer.zero_grad()
            outputs = self.model(x_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """
        Evaluate the model on validation data.
        
        Parameters
        ----------
        val_loader : torch.utils.data.DataLoader
            DataLoader containing validation data.
            
        Returns
        -------
        float
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x_batch, y_batch = [b.to(self.device) for b in batch]
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
        return total_loss / len(val_loader)
    
    def evaluate(self, test_loader):
        """
        Evaluate the model on test data.
        
        Parameters
        ----------
        test_loader : torch.utils.data.DataLoader
            DataLoader containing test data.
            
        Returns
        -------
        dict
            Dictionary containing test metrics.
        """
        self.model.eval()
        total_loss = 0
        predictions = []
        ground_truth = []
        
        with torch.no_grad():
            for batch in test_loader:
                x_batch, y_batch = [b.to(self.device) for b in batch]
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
                
                # Store predictions and ground truth
                predictions.extend(outputs.cpu().numpy())
                ground_truth.extend(y_batch.cpu().numpy())
        
        # Calculate metrics
        mse = total_loss / len(test_loader)
        mae = np.mean(np.abs(np.array(predictions) - np.array(ground_truth)))
        
        return {
            "test_mse": mse,
            "test_mae": mae,
            "test_rmse": np.sqrt(mse)
        }
    
    def train(self, data_loaders, epochs=1000, patience=10, model_dir='models', experiment_name=None):
        """
        Train the model with early stopping and learning rate scheduling.
        
        Parameters
        ----------
        data_loaders : dict
            Dictionary containing DataLoader objects for 'train' and 'val'.
        epochs : int, default=1000
            Maximum number of epochs to train.
        patience : int, default=10
            Number of epochs with no improvement after which training will be stopped.
        model_dir : str, default='models'
            Directory to save trained models.
        experiment_name : str, optional
            Name for the experiment. Used in the saved model filename.
            
        Returns
        -------
        dict
            Dictionary containing training history.
        """
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Generate model filename
        if experiment_name is None:
            experiment_name = f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model_filename = os.path.join(model_dir, f"{experiment_name}_best.pth")
        
        best_val_loss = float('inf')
        early_stop_count = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(data_loaders['train'])
            val_loss = self.validate(data_loaders['val'])
            self.scheduler.step(val_loss)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.8f}")
            
            self.validation_loss_history.append(val_loss)
            self.training_loss_history.append(train_loss)
            self.epoch_history.append(epoch)
            self.lr_history.append(current_lr)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_count = 0
            else:
                early_stop_count += 1
                
            if early_stop_count >= patience:
                print("Early stopping triggered!")
                break
        
        # Evaluate on test data if available
        test_metrics = None
        if 'test' in data_loaders:
            test_metrics = self.evaluate(data_loaders['test'])
            print(f"Test metrics: {test_metrics}")
        
        # Create training history compatible with json package
        training_history = {
            "epochs": self.epoch_history,
            "train_loss": [float(loss) for loss in self.training_loss_history],  # Convert all to float
            "val_loss": [float(loss) for loss in self.validation_loss_history],  # Convert all to float
            "learning_rate": [float(lr) for lr in self.lr_history],  # Convert all to float
            "best_val_loss": float(best_val_loss),  # Convert to float
            "stopped_epoch": int(epoch + 1)  # Ensure integer type
        }

        # Log experiment if logger is available
        if self.logger is not None:
            val_metrics = {"val_loss": float(best_val_loss)}  # Convert to Python float
            test_metrics = {k: float(v) for k, v in test_metrics.items()} if test_metrics else None  # Convert test metrics

            log_path = self.logger.log_experiment(
                self.model_config,
                training_history,
                val_metrics=val_metrics,
                test_metrics=test_metrics
            )
            print(f"Experiment logged to {log_path}")

        
        return training_history
    
    #TODO test this out and see if its faster than the current method in monte_carlo_sim.py
    def forecast(self, initial_sequence, num_steps, temperature=0.1, num_samples=100):
        """
        Generate forecasts using the trained model.
        
        Parameters
        ----------
        initial_sequence : torch.Tensor or numpy.ndarray
            Initial sequence to start forecasting from
        num_steps : int
            Number of steps to forecast into the future
        temperature : float, default=0.1
            Base temperature for uncertainty in predictions
        num_samples : int, default=100
            Number of Monte Carlo samples to generate
            
        Returns
        -------
        dict
            Dictionary containing forecast results:
            - 'mean': Mean predictions
            - 'std': Standard deviation of predictions
            - 'samples': All Monte Carlo samples if num_samples > 1
            
        Notes
        -----
        This method handles converting the initial sequence to the correct format
        and device, then uses the model's generate_forecast method to produce
        predictions with uncertainty quantification.
        """
        self.model.eval()
        
        # Convert initial sequence to tensor if needed
        if not isinstance(initial_sequence, torch.Tensor):
            initial_sequence = torch.tensor(initial_sequence, dtype=torch.float32)
        
        # Add batch dimension if needed
        if initial_sequence.dim() == 2:
            initial_sequence = initial_sequence.unsqueeze(0)
        
        # Move to correct device
        initial_sequence = initial_sequence.to(self.device)
        
        # Generate forecast
        forecast_results = self.model.generate_forecast(
            initial_sequence=initial_sequence,
            num_steps=num_steps,
            temperature=temperature,
            num_samples=num_samples
        )
        
        return forecast_results

#TODO Have a json of configs to reload to if an interuption occurs
# This json will have a list of all remaining parameters to run in and only clear 
def run_decoder_tuning(data, sequence_size=30, d_model=512, nhead=8, num_layers=7, 
                      dropout=0.2, learning_rate=1e-4, epochs=50, patience=10, 
                      plot_training=False, loss_type='mse'):
    """
    Run comprehensive hyperparameter tuning for decoder architectures.

    This function performs systematic evaluation of different decoder configurations
    for transformer models. It handles the complete tuning process including
    data preparation, model creation, training, evaluation, and result logging.
    It implements checkpointing to handle interruptions gracefully.

    Parameters
    ----------
    data : pandas.DataFrame
        Input time series data with features and a 'Date' column.
    sequence_size : int, default=30
        Length of input sequences for the model.
    d_model : int, default=512
        Dimension of the transformer model's embeddings.
    nhead : int, default=8
        Number of attention heads in transformer layers.
    num_layers : int, default=7
        Number of transformer encoder layers.
    dropout : float, default=0.2
        Dropout rate for transformer layers.
    learning_rate : float, default=1e-4
        Initial learning rate for optimization.
    epochs : int, default=50
        Maximum number of training epochs per configuration.
    patience : int, default=10
        Number of epochs without improvement before early stopping.
    plot_training : bool, default=False
        Whether to plot training history after each configuration trains.
    loss_type : str, default='custom'
        Type of loss function to use:
        - 'custom': Uses CustomLoss with trend, volatility, and momentum preservation
        - 'asymptote_prevention': Uses AsymptotePreventionLoss to prevent convergence
        - 'forecast_based': Uses ForecastBasedLoss to evaluate prediction quality using forecasts
        - 'mse': Uses standard MSE loss

    Returns
    -------
    dict
        Dictionary containing:
        
        - experiment_path : str
            Path to the experiment directory
        - top_models : list
            List of the 5 best performing model configurations and their metrics

    Notes
    -----
    Experimental Design:
        
    1. Directory Structure
        - Creates timestamped experiment directories
        - Maintains separate directories for models and logs
        - Saves checkpoints for interruption recovery
    
    2. Model Selection
        - Uses min-heap to track top 5 performing models based on training loss
        - Saves both model states and configurations for later use
        - All floating-point values are rounded to 4 decimal places
    
    3. Performance Optimization
        - Implements early stopping for efficient training
        - Uses GPU acceleration when available
        - Manages memory through periodic cleanup
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TimeSeriesDataset(sequence_size=sequence_size)
    data_loaders = dataset.prepare_data(data)
    
    # Calculate historical range for loss functions that need it
    price_data = data['Price'].values
    historical_range = (float(np.min(price_data)), float(np.max(price_data)))
    
    # Configure loss function based on type
    loss_config = {
        'type': loss_type
    }
    
    # Add parameters based on loss type
    if loss_type == 'asymptote_prevention':
        loss_config.update({
            'historical_range': historical_range,
            'max_daily_change': 0.1
        })
    elif loss_type == 'custom':
        loss_config.update({
            'alpha': 0.4,
            'beta': 0.3,
            'gamma': 0.3
        })
    elif loss_type == 'forecast_based':
        loss_config.update({
            'forecast_steps': 5,
            'forecast_weight': 0.4,
            'temperature': 0.1,
            'num_samples': 10
        })
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_path = Path("experiments") / f"experiment_{timestamp}"
    model_best_path = experiment_path / "best_models"
    experiment_path.mkdir(parents=True, exist_ok=True)
    model_best_path.mkdir(parents=True, exist_ok=True)
    
    logger = ModelTuningLogger(log_dir=experiment_path)
    decoder_configs = create_decoder_grid(d_model)
    
    top_models = []
    training_losses = []
    checkpoint_file = experiment_path / "checkpoint.json"
    
    print(f"\nInitializing training with loss configuration:")
    print(f"{'='*80}")
    print(f"Loss type: {loss_type}")
    if loss_type == 'asymptote_prevention':
        print(f"Historical range: [{historical_range[0]:.4f}, {historical_range[1]:.4f}]")
    for key, value in loss_config.items():
        if key not in ['type', 'historical_range']:
            print(f"{key}: {value}")
    print(f"{'='*80}\n")
    
    for i, config in zip(range(0,len(decoder_configs)), decoder_configs):
        print(f'Model {i+1} of {len(decoder_configs)}: {(i+1)/len(decoder_configs):.2%} done')
        model = TransformerModel(
            input_dim=len(data.columns) - 1,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            decoder_config=config
        ).to(device)
        
        # Print detailed model configuration
        print(f"\n{'='*80}")
        print(f"TRAINING MODEL WITH CONFIGURATION:")
        print(f"{'='*80}")
        print(f"Base parameters:")
        print(f"  - Input dimensions: {len(data.columns) - 1}")
        print(f"  - Embedding dimension (d_model): {d_model}")
        print(f"  - Attention heads (nhead): {nhead}")
        print(f"  - Transformer layers: {num_layers}")
        print(f"  - Dropout rate: {dropout:.4f}")
        print(f"  - Learning rate: {learning_rate:.4f}")
        print(f"\nDecoder configuration:")
        print(f"  - Layer architecture: {[(size, act) for size, act in config['hidden_layers']]}")
        print(f"  - Batch normalization: {config['use_batch_norm']}")
        print(f"  - Decoder dropout: {config.get('dropout', 0):.4f}")
        print(f"{'='*80}\n")
        
        trainer = ModelTrainer(
            model, 
            device, 
            learning_rate, 
            model_config=config, 
            logger=logger,
            loss_config=loss_config
        )
        
        training_history = trainer.train(data_loaders, epochs=epochs, patience=patience, model_dir=model_best_path)
        
        if plot_training:
            print("\nTraining History Plot:")
            plot_training_history(trainer)
        
        # Get minimum training loss from history and round to 4 decimal places
        train_loss = round(min(training_history["train_loss"]), 4)
        val_loss = round(training_history["best_val_loss"], 4)
        
        model_info = {
            'model_state_dict': model.state_dict(),
            'config': config,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'loss_config': loss_config  # Store the complete loss configuration
        }
        
        # Push to heap based on training loss
        heapq.heappush(top_models, (train_loss, model_info))
        if len(top_models) > 5:
            heapq.heappop(top_models)
        
        training_losses.append({
            'config': config,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'loss_config': loss_config
        })
        
        try:
            serializable_losses = uf.convert_to_serializable(training_losses)
            with open(checkpoint_file, 'w') as f:
                json.dump(serializable_losses, f, indent=4)
        except TypeError as e:
            print(f"Warning: JSON serialization error with training losses: {e}")
    
    # Save top 5 models based on training loss
    for i, (train_loss, model_info) in enumerate(sorted(top_models)):
        model_path = experiment_path / f'best_model_{i+1}_train_loss_{train_loss:.4f}.pth'
        torch.save(model_info['model_state_dict'], model_path)
        
        # Save corresponding configuration
        config_path = experiment_path / f'best_model_{i+1}_config.json'
        config_info = {
            'model_config': model_info['config'],
            'loss_config': model_info['loss_config'],
            'train_loss': train_loss,
            'val_loss': model_info['val_loss']
        }
        with open(config_path, 'w') as f:
            json.dump(uf.convert_to_serializable(config_info), f, indent=4)
    
    # Convert top models to serializable format with rounded values
    top_models_json = [{
        "config": model[1]["config"],
        "loss_config": model[1]["loss_config"],
        "train_loss": round(model[0], 4),
        "val_loss": round(model[1]["val_loss"], 4)
    } for model in top_models]
    top_models_json = uf.convert_to_serializable(top_models_json)
    
    try:
        with open(experiment_path / "top_models.json", 'w') as f:
            json.dump(top_models_json, f, indent=4)
    except TypeError as e:
        print(f"Warning: JSON serialization error with top model configurations: {e}")
    
    return {"experiment_path": str(experiment_path), "top_models": top_models_json}


#TODO Use better names for variable here
class CustomLoss(nn.Module):
    """
    Custom loss function for time series forecasting that combines multiple objectives:
    1. MSE for basic prediction accuracy
    2. Trend preservation loss to maintain price movement direction
    3. Volatility matching loss to maintain price variation patterns
    4. Momentum preservation loss to prevent asymptote convergence
    
    Parameters
    ----------
    alpha : float, default=0.4
        Weight for trend preservation loss
    beta : float, default=0.3
        Weight for volatility matching loss
    gamma : float, default=0.3
        Weight for momentum preservation loss
    """
    def __init__(self, alpha=0.4, beta=0.3, gamma=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse = nn.MSELoss()
        
    def forward(self, predictions, targets):
        # Basic MSE loss
        mse_loss = self.mse(predictions, targets)
        
        # Trend preservation loss
        pred_diff = predictions[1:] - predictions[:-1]
        target_diff = targets[1:] - targets[:-1]
        trend_loss = torch.mean(torch.abs(torch.sign(pred_diff) - torch.sign(target_diff)))
        
        # Volatility matching loss
        pred_vol = torch.std(predictions)
        target_vol = torch.std(targets)
        vol_loss = torch.abs(pred_vol - target_vol)
        
        # Momentum preservation loss (prevents asymptote convergence)
        pred_momentum = predictions[2:] - 2 * predictions[1:-1] + predictions[:-2]
        target_momentum = targets[2:] - 2 * targets[1:-1] + targets[:-2]
        momentum_loss = torch.mean(torch.abs(pred_momentum - target_momentum))
        
        # Combine losses
        total_loss = (1 - self.alpha - self.beta - self.gamma) * mse_loss + \
                    self.alpha * trend_loss + \
                    self.beta * vol_loss + \
                    self.gamma * momentum_loss
        
        return total_loss

class AsymptotePreventionLoss(nn.Module):
    """
    Loss function specifically designed to prevent asymptote convergence by:
    1. Penalizing predictions that deviate too far from the historical range
    2. Encouraging price movement continuation
    3. Maintaining realistic price changes
    
    Parameters
    ----------
    historical_range : tuple
        (min_price, max_price) from historical data
    max_daily_change : float
        Maximum allowed daily price change as a percentage
    """
    def __init__(self, historical_range, max_daily_change=0.1):
        super().__init__()
        self.min_price, self.max_price = historical_range
        self.max_daily_change = max_daily_change
        self.mse = nn.MSELoss()
        
    def forward(self, predictions, targets):
        # Basic MSE loss
        mse_loss = self.mse(predictions, targets)
        
        # Range violation penalty
        range_violation = torch.mean(torch.relu(predictions - self.max_price) + 
                          torch.relu(self.min_price - predictions))
        
        # Daily change violation penalty
        daily_changes = torch.abs(predictions[1:] - predictions[:-1]) / predictions[:-1]
        change_violation = torch.mean(torch.relu(daily_changes - self.max_daily_change))
        
        # Combine losses
        total_loss = mse_loss + 0.5 * range_violation + 0.3 * change_violation
        
        return total_loss

class AdaptiveForecastLoss(nn.Module):
    """
    DEPRECATED severly broken
    Advanced loss function that incorporates multi-step forecasting and learnable parameters
    for dynamic adaptation to market conditions.
    
    This loss function:
    1. Performs multi-step forecasting within the loss calculation
    2. Uses learnable parameters to adapt penalties
    3. Implements dynamic scaling based on forecast horizon
    4. Combines multiple objectives with adaptive weights
    
    Parameters
    ----------
    historical_range : tuple
        (min_price, max_price) from historical data
    forecast_steps : int, default=5
        Number of future steps to consider in forecasting component
    init_max_change : float, default=0.1
        Initial value for maximum daily change parameter
    init_range_weight : float, default=0.5
        Initial weight for range violation penalty
    init_change_weight : float, default=0.3
        Initial weight for change violation penalty
    """
    def __init__(self, historical_range, forecast_steps=5, init_max_change=0.1,
                 init_range_weight=0.5, init_change_weight=0.3):
        super().__init__()
        # Create tensors on CPU initially
        self.register_buffer('min_price', torch.tensor(float(historical_range[0]), dtype=torch.float32))
        self.register_buffer('max_price', torch.tensor(float(historical_range[1]), dtype=torch.float32))
        self.register_buffer('price_scale', torch.abs(torch.tensor(float(historical_range[1]) - float(historical_range[0]), 
                                                                  dtype=torch.float32)).clamp(min=1e-6))
        
        self.forecast_steps = min(forecast_steps, 5)  # Limit forecast steps
        
        # Initialize learnable parameters with more stable values
        self.max_change = nn.Parameter(torch.tensor(init_max_change, dtype=torch.float32).clamp(0.01, 0.5))
        self.range_weight = nn.Parameter(torch.tensor(init_range_weight, dtype=torch.float32).clamp(0.1, 0.9))
        self.change_weight = nn.Parameter(torch.tensor(init_change_weight, dtype=torch.float32).clamp(0.1, 0.9))
        
        # Initialize horizon scaling with more conservative values
        initial_scales = torch.exp(torch.arange(self.forecast_steps, dtype=torch.float32) * -0.2)
        self.horizon_scale = nn.Parameter(initial_scales.clamp(0.1, 1.0))
        
        # Base loss functions
        self.mse = nn.MSELoss(reduction='mean')
        
        # Constants for numerical stability
        self.eps = 1e-6
    
    def to(self, device):
        """Override to method to handle device placement for all tensors"""
        super().to(device)
        # Move all tensors to the specified device
        self.min_price = self.min_price.to(device)
        self.max_price = self.max_price.to(device)
        self.price_scale = self.price_scale.to(device)
        self.max_change = self.max_change.to(device)
        self.range_weight = self.range_weight.to(device)
        self.change_weight = self.change_weight.to(device)
        self.horizon_scale = self.horizon_scale.to(device)
        self.mse = self.mse.to(device)
        return self
        
    def forecast_loss(self, predictions, targets):
        """Calculate loss across multiple forecast steps with increasing uncertainty"""
        device = predictions.device
        total_forecast_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        batch_size = predictions.shape[0]
        
        # Ensure all tensors are on the same device
        min_price = self.min_price.to(device)
        max_price = self.max_price.to(device)
        
        # Early return if batch is too small
        if batch_size < 2:
            return self.mse(predictions, targets)
        
        # Calculate immediate prediction loss with full weight
        immediate_loss = self.mse(predictions.clamp(min_price, max_price), 
                                targets.clamp(min_price, max_price))
        total_forecast_loss += immediate_loss
        
        # Calculate multi-step forecast loss with increasing uncertainty
        valid_steps = min(self.forecast_steps, batch_size - 1)
        if valid_steps > 0:
            for step in range(1, valid_steps + 1):
                # Get prediction and target for this step
                step_pred = predictions[:-step].clamp(min_price, max_price)
                step_target = targets[step:].clamp(min_price, max_price)
                
                if len(step_pred) > 0:
                    # Calculate base step loss
                    step_loss = self.mse(step_pred, step_target)
                    
                    # Apply decreasing weight for longer horizons
                    scale = torch.sigmoid(self.horizon_scale[step-1].to(device)).clamp(0.1, 1.0)
                    total_forecast_loss += scale * step_loss
            
            # Normalize with a stable denominator
            total_forecast_loss = total_forecast_loss / (valid_steps + 1)
        
        return total_forecast_loss
    
    def adaptive_range_penalty(self, predictions):
        """Calculate adaptive range violation penalty with increasing bounds"""
        device = predictions.device
        
        # Ensure all tensors are on the same device
        min_price = self.min_price.to(device)
        max_price = self.max_price.to(device)
        price_scale = self.price_scale.to(device)
        
        # Calculate normalized predictions with clamping
        predictions = predictions.clamp(min_price - price_scale, max_price + price_scale)
        norm_predictions = (predictions - min_price) / (price_scale + self.eps)
        
        # Expanding boundaries over time with stability
        sequence_pos = torch.arange(predictions.shape[0], device=device, dtype=torch.float32)
        sequence_pos = sequence_pos / (predictions.shape[0] + self.eps)
        expansion_factor = 1.0 + sequence_pos.unsqueeze(1)
        
        # Soft boundaries with expanding margins
        margin = 0.1 * expansion_factor.clamp(1.0, 2.0)
        upper_violation = torch.relu(norm_predictions - (1 + margin))
        lower_violation = torch.relu(-norm_predictions + (-margin))
        
        # Combine violations with reduced penalty for later timesteps
        penalty = torch.mean((upper_violation + lower_violation) / (expansion_factor + self.eps))
        return penalty.clamp(0.0, 10.0)  # Prevent extreme penalties
    
    def adaptive_change_penalty(self, predictions):
        """Calculate adaptive change penalty with increasing allowance"""
        device = predictions.device
        
        # Ensure all tensors are on the same device
        min_price = self.min_price.to(device)
        max_price = self.max_price.to(device)
        price_scale = self.price_scale.to(device)
        max_change = self.max_change.to(device)
        
        # Normalize predictions with clamping
        predictions = predictions.clamp(min_price, max_price)
        norm_predictions = predictions / (price_scale + self.eps)
        
        # Calculate relative changes
        diffs = norm_predictions[1:] - norm_predictions[:-1]
        
        # Allow larger changes over time with stability
        sequence_pos = torch.arange(diffs.shape[0], device=device, dtype=torch.float32)
        sequence_pos = sequence_pos / (diffs.shape[0] + self.eps)
        max_change_val = torch.sigmoid(max_change).clamp(0.01, 0.5) * (0.2 + 0.3 * sequence_pos.unsqueeze(1))
        
        # Calculate penalty with increasing tolerance
        changes = torch.abs(diffs).clamp(0.0, 1.0)
        penalty = torch.mean(torch.relu(changes - max_change_val) / (1 + sequence_pos.unsqueeze(1) + self.eps))
        return penalty.clamp(0.0, 10.0)  # Prevent extreme penalties
    
    def momentum_continuity(self, predictions):
        """Calculate momentum with increasing flexibility"""
        device = predictions.device
        if predictions.shape[0] < 3:
            return torch.tensor(0.0, device=device, dtype=torch.float32)
        
        # Ensure all tensors are on the same device
        min_price = self.min_price.to(device)
        max_price = self.max_price.to(device)
        price_scale = self.price_scale.to(device)
        
        # Normalize predictions with clamping
        predictions = predictions.clamp(min_price, max_price)
        norm_predictions = predictions / (price_scale + self.eps)
        
        # Calculate second-order differences with increasing tolerance
        momentum = norm_predictions[2:] - 2 * norm_predictions[1:-1] + norm_predictions[:-2]
        
        # Allow more momentum variation over time with stability
        sequence_pos = torch.arange(momentum.shape[0], device=device, dtype=torch.float32)
        sequence_pos = sequence_pos / (momentum.shape[0] + self.eps)
        tolerance = (1.0 + sequence_pos.unsqueeze(1)).clamp(1.0, 3.0)
        
        # Reduced penalty for later timesteps
        penalty = torch.mean(torch.abs(momentum).clamp(0.0, 1.0) / (tolerance + self.eps))
        return penalty.clamp(0.0, 10.0)  # Prevent extreme penalties
    
    def forward(self, predictions, targets):
        """Calculate total loss with natural uncertainty growth"""
        device = predictions.device
        predictions = predictions.to(device, dtype=torch.float32)
        targets = targets.to(device, dtype=torch.float32)
        
        # Ensure all parameters are on the correct device
        range_weight = self.range_weight.to(device)
        change_weight = self.change_weight.to(device)
        
        # Base prediction loss with clamping
        forecast_loss = self.forecast_loss(predictions, targets)
        
        # Calculate penalties with learnable weights and stability
        range_w = torch.sigmoid(range_weight).clamp(0.1, 0.5)
        change_w = torch.sigmoid(change_weight).clamp(0.1, 0.5)
        
        # Calculate penalties
        range_penalty = self.adaptive_range_penalty(predictions)
        change_penalty = self.adaptive_change_penalty(predictions)
        momentum_penalty = self.momentum_continuity(predictions)
        
        # Combine components with stability
        total_loss = (
            forecast_loss +  # Main prediction loss
            range_w * range_penalty +  # Flexible range constraints
            change_w * change_penalty +  # Adaptive change allowance
            0.1 * momentum_penalty  # Light momentum guidance
        )
        
        # Final stability check
        return total_loss.clamp(0.0, 1e6)  # Prevent infinite loss values

class DirectionalLoss(nn.Module):
    """
    A simple loss function that combines MSE with cosine similarity to ensure
    predictions follow the correct directional trend.
    
    This loss function:
    1. Uses normalized MSE (bounded between 0 and 1) for point-wise accuracy
    2. Uses cosine similarity to ensure directional alignment
    3. Combines both metrics with a weighted sum
    
    Parameters
    ----------
    direction_weight : float, default=0.3
        Weight given to the directional (cosine) component of the loss.
        The MSE component will have weight (1 - direction_weight).
    mse_scale : float, default=10.0
        Scale factor for MSE normalization. Higher values make the sigmoid
        transition sharper around smaller MSE values.
    """
    def __init__(self, direction_weight=0.3, mse_scale=10.0):
        super().__init__()
        self.direction_weight = direction_weight
        self.mse_scale = mse_scale
        self.mse = nn.MSELoss()
        self.cos = nn.CosineSimilarity(dim=0)
        
    def forward(self, predictions, targets):
        # Ensure inputs are on the same device
        device = predictions.device
        predictions = predictions.to(device)
        targets = targets.to(device)
        
        # Calculate MSE component and normalize it to [0, 1]
        mse_loss = self.mse(predictions, targets)
        # Use sigmoid to bound MSE between 0 and 1
        # Scale MSE to make sigmoid more sensitive to relevant error ranges
        normalized_mse = torch.sigmoid(self.mse_scale * mse_loss)
        
        # Calculate directional component using cosine similarity
        # Reshape predictions and targets to 1D for cosine similarity
        pred_flat = predictions.view(-1)
        target_flat = targets.view(-1)
        
        # Cosine similarity returns 1 for identical directions, -1 for opposite
        # Convert to a loss (0 for identical, 1 for opposite)
        direction_loss = (1 - self.cos(pred_flat, target_flat)) / 2
        
        # Combine losses with weights (both components now bounded [0, 1])
        total_loss = (1 - self.direction_weight) * normalized_mse + \
                    self.direction_weight * direction_loss
        
        return total_loss

class ForecastBasedLoss(nn.Module):
    """
    Enhanced loss function that combines:
    1. Directional accuracy using cosine similarity
    2. Bounded MSE for point accuracy
    3. Volatility matching for realistic price movements
    4. Uncertainty-weighted multi-step forecasting
    
    Parameters
    ----------
    model : TransformerModel
        The model being trained, used to generate forecasts
    forecast_steps : int, default=5
        Number of steps to forecast for loss calculation
    forecast_weight : float, default=0.4
        Weight given to forecast-based loss vs immediate prediction loss
    temperature : float, default=0.1
        Temperature for forecast uncertainty
    num_samples : int, default=10
        Number of Monte Carlo samples for each forecast
    mse_scale : float, default=10.0
        Scale factor for MSE normalization in sigmoid
    direction_weight : float, default=0.4
        Weight for directional component in forecast evaluation
    volatility_weight : float, default=0.2
        Weight for volatility matching component
    """
    def __init__(self, model, forecast_steps=5, forecast_weight=0.4, temperature=0.1, 
                 num_samples=10, mse_scale=10.0, direction_weight=0.4, volatility_weight=0.2):
        super().__init__()
        self.model = model
        self.forecast_steps = forecast_steps
        self.forecast_weight = forecast_weight
        self.temperature = temperature
        self.num_samples = num_samples
        self.mse_scale = mse_scale
        self.direction_weight = direction_weight
        self.volatility_weight = volatility_weight
        
        # Loss functions
        self.mse = nn.MSELoss(reduction='none')
        self.cos = nn.CosineSimilarity(dim=0)
        
    def calculate_step_losses(self, predictions, targets):
        """
        Calculate all loss components for a single prediction step.
        All components are bounded between 0 and 1 using sigmoid for equal weighting.
        """
        device = predictions.device
        
        # Bounded MSE loss
        mse_loss = self.mse(predictions, targets).mean()
        normalized_mse = torch.sigmoid(self.mse_scale * mse_loss)
        
        # Directional loss using cosine similarity
        pred_flat = predictions.view(-1)
        target_flat = targets.view(-1)
        # Cosine similarity is already bounded [-1, 1], convert to [0, 1]
        direction_loss = (1 - self.cos(pred_flat, target_flat)) / 2
        
        # Volatility matching loss - normalize using sigmoid
        if len(predictions) > 1:
            pred_volatility = torch.std(predictions)
            target_volatility = torch.std(targets)
            # Scale relative volatility difference before sigmoid
            rel_vol_diff = torch.abs(pred_volatility - target_volatility) / (target_volatility + 1e-6)
            volatility_loss = torch.sigmoid(self.mse_scale * rel_vol_diff)
        else:
            volatility_loss = torch.tensor(0.0, device=device)
            
        # Momentum matching loss (new component)
        if len(predictions) > 2:
            pred_momentum = predictions[2:] - 2 * predictions[1:-1] + predictions[:-2]
            target_momentum = targets[2:] - 2 * targets[1:-1] + targets[:-2]
            momentum_diff = torch.mean(torch.abs(pred_momentum - target_momentum))
            # Normalize momentum difference
            momentum_loss = torch.sigmoid(self.mse_scale * momentum_diff)
        else:
            momentum_loss = torch.tensor(0.0, device=device)
        
        return {
            'mse': normalized_mse,
            'direction': direction_loss,
            'volatility': volatility_loss,
            'momentum': momentum_loss
        }
        
    def generate_forecast_sequence(self, input_sequences):
        """
        Generate forecasts for each sequence in the batch.
        
        Parameters
        ----------
        input_sequences : torch.Tensor
            Batch of input sequences [batch_size, seq_length, features]
            
        Returns
        -------
        tuple
            (forecasts, forecast_stds) where:
            - forecasts: tensor of shape [batch_size, forecast_steps]
            - forecast_stds: tensor of shape [batch_size, forecast_steps]
        """
        device = input_sequences.device
        batch_size = input_sequences.shape[0]
        
        # Initialize storage for forecasts and their uncertainties
        forecasts = torch.zeros((batch_size, self.forecast_steps), device=device)
        forecast_stds = torch.zeros((batch_size, self.forecast_steps), device=device)
        
        # Generate forecasts for each sequence
        for i in range(batch_size):
            sequence = input_sequences[i:i+1]  # Keep batch dimension
            
            # Generate forecast using model's method
            forecast_results = self.model.generate_forecast(
                initial_sequence=sequence,
                num_steps=self.forecast_steps,
                temperature=self.temperature,
                num_samples=self.num_samples
            )
            
            # Convert numpy arrays to tensors and store
            forecasts[i] = torch.tensor(forecast_results['mean'], device=device)
            forecast_stds[i] = torch.tensor(forecast_results['std'], device=device)
        
        return forecasts, forecast_stds
    
    def forward(self, predictions, targets):
        """
        Calculate combined loss using immediate predictions and generated forecasts.
        All components are sigmoid-bounded and properly weighted.
        """
        device = predictions.device
        batch_size = predictions.shape[0]
        
        # Calculate immediate prediction losses
        immediate_losses = self.calculate_step_losses(predictions, targets)
        
        # Combine immediate losses with equal weighting in [0, 1] range
        immediate_loss = (
            (1 - self.direction_weight - self.volatility_weight) * immediate_losses['mse'] +
            self.direction_weight * immediate_losses['direction'] +
            self.volatility_weight * (
                0.7 * immediate_losses['volatility'] + 
                0.3 * immediate_losses['momentum']  # Add momentum component
            )
        )
        
        # Get sequences from the model's last input
        if not hasattr(self.model, 'last_input_sequence'):
            return immediate_loss
        
        input_sequences = self.model.last_input_sequence
        
        # Generate forecasts for each sequence
        forecasts, forecast_stds = self.generate_forecast_sequence(input_sequences)
        
        # Calculate forecast-based loss components
        forecast_losses = []
        
        # For each step we can evaluate (limited by available targets)
        valid_steps = min(self.forecast_steps, targets.shape[0] - 1)
        if valid_steps > 0:
            for step in range(valid_steps):
                # Get the relevant predictions and targets for this step
                step_forecasts = forecasts[:-step-1, step]
                step_targets = targets[step+1:]
                step_stds = forecast_stds[:-step-1, step]
                
                if len(step_forecasts) > 0:
                    # Calculate all loss components for this step
                    step_losses = self.calculate_step_losses(
                        step_forecasts, 
                        step_targets.squeeze()
                    )
                    
                    # Combine step losses with equal weighting
                    step_loss = (
                        (1 - self.direction_weight - self.volatility_weight) * step_losses['mse'] +
                        self.direction_weight * step_losses['direction'] +
                        self.volatility_weight * (
                            0.7 * step_losses['volatility'] + 
                            0.3 * step_losses['momentum']  # Add momentum component
                        )
                    )
                    
                    # Weight by uncertainty (also sigmoid bounded)
                    uncertainty_weights = torch.sigmoid(-step_stds)  # Higher std = lower weight
                    weighted_loss = (step_loss * uncertainty_weights).mean()
                    forecast_losses.append(weighted_loss)
        
        # If we have forecast losses, combine them with immediate loss
        if forecast_losses:
            forecast_loss = torch.stack(forecast_losses).mean()
            # Both components are already bounded [0, 1], so their weighted sum will be too
            total_loss = (1 - self.forecast_weight) * immediate_loss + self.forecast_weight * forecast_loss
        else:
            total_loss = immediate_loss
        
        return total_loss
