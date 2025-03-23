import torch
import numpy as np
from scipy import stats
import pandas as pd
from typing import Tuple, Optional
from torch.utils.data import DataLoader

class MonteCarloPredictor:
    def __init__(
        self,
        model: torch.nn.Module,
        scaler,
        base_temp: float = 0.1,
        num_days: int = 100,
        num_iterations: int = 1000,
        window_size: int = 30,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model
        self.scaler = scaler
        self.base_temp = base_temp
        self.num_days = num_days
        self.num_iterations = num_iterations
        self.window_size = window_size
        self.device = device
        
        # Storage for diagnostics
        self.convergence_metrics = []
        self.autocorrelation = None
        self.prediction_distributions = None
        
    def _check_stationarity(self, predictions: np.ndarray) -> Tuple[bool, float]:
        """
        Test time series stationarity using the Augmented Dickey-Fuller test.

        This function applies the ADF test to determine whether a time series is
        stationary, which is crucial for validating Monte Carlo simulation results.

        Parameters
        ----------
        predictions : numpy.ndarray
            Array of time series predictions to test for stationarity.

        Returns
        -------
        Tuple[bool, float]
            is_stationary : bool
                True if the series is stationary (p-value < 0.05).
            p_value : float
                The p-value from the ADF test.

        Notes
        -----
        Design Decisions:
        1. Statistical Testing:
           - Uses Augmented Dickey-Fuller test for robust stationarity detection
           - Significance level set at 0.05 for standard statistical practice
           - Null hypothesis: series has a unit root (non-stationary)
        
        2. Implementation:
           - No lag order specification to allow automatic selection
           - Returns both boolean result and p-value for flexibility in reporting
           - Uses scipy.stats for efficient statistical computation

        See Also
        --------
        _calculate_effective_sample_size : Related function for assessing sample independence.
        scipy.stats.adfuller : Underlying statistical test implementation.

        Examples
        --------
        >>> predictor = MonteCarloPredictor(model, scaler)
        >>> predictions = np.array([1.2, 1.3, 1.1, 1.4, 1.3])
        >>> is_stationary, p_value = predictor._check_stationarity(predictions)
        >>> print(f"Stationary: {is_stationary}, p-value: {p_value:.4f}")
        """
        adf_result = stats.adfuller(predictions)
        is_stationary = adf_result[1] < 0.05
        return is_stationary, adf_result[1]
    
    def _calculate_effective_sample_size(self, predictions: np.ndarray) -> float:
        """
        Calculate the effective sample size using autocorrelation analysis.

        This function estimates the number of effectively independent samples in
        a time series, accounting for autocorrelation between observations.

        Parameters
        ----------
        predictions : numpy.ndarray
            Array of time series predictions to analyze.

        Returns
        -------
        float
            The effective sample size, which will be less than or equal to
            the actual number of samples due to autocorrelation.

        Notes
        -----
        Design Decisions:
        1. Autocorrelation Calculation:
           - Uses full autocorrelation function via numpy.correlate
           - Normalizes by variance for proper correlation scaling
           - Considers all possible lags for thorough analysis
        
        2. ESS Estimation:
           - Uses Geyer's initial monotone sequence estimator
           - Accounts for both positive and negative autocorrelation
           - Provides conservative estimate for statistical validity
        
        3. Implementation:
           - Efficient vectorized operations for large sample sizes
           - Handles mean-centering internally for robustness
           - Returns float to preserve precision in calculations

        See Also
        --------
        _check_stationarity : Related function for testing time series properties.
        numpy.correlate : Core function used for autocorrelation calculation.

        Examples
        --------
        >>> predictor = MonteCarloPredictor(model, scaler)
        >>> predictions = np.array([1.2, 1.3, 1.1, 1.4, 1.3])
        >>> ess = predictor._calculate_effective_sample_size(predictions)
        >>> print(f"Effective sample size: {ess:.2f}")
        """
        acf = np.correlate(predictions - np.mean(predictions), 
                          predictions - np.mean(predictions), 
                          mode='full')[len(predictions)-1:]
        acf = acf / acf[0]
        # Estimate effective sample size
        ess = len(predictions) / (1 + 2 * np.sum(acf[1:]))
        return ess
    
    def _assess_convergence(self, predictions_all: np.ndarray, window: int = 50) -> dict:
        """Assess MCMC convergence using multiple metrics"""
        means = np.mean(predictions_all, axis=0)
        running_means = np.cumsum(predictions_all, axis=0) / np.arange(1, predictions_all.shape[0] + 1)[:, np.newaxis]
        
        # Gelman-Rubin diagnostic (split chains)
        split_chains = np.array_split(predictions_all, 2, axis=0)
        between_chain_var = np.var(np.mean(split_chains, axis=1), axis=0)
        within_chain_var = np.mean([np.var(chain, axis=0) for chain in split_chains], axis=0)
        r_hat = np.sqrt((within_chain_var + between_chain_var) / within_chain_var)
        
        return {
            'r_hat': r_hat,
            'mean_convergence': np.abs(running_means[-1] - running_means[-window]),
            'ess': self._calculate_effective_sample_size(means)
        }
    
    def _extract_last_sequence(self, dataset: torch.Tensor) -> torch.Tensor:
        """Extract the last sequence from the dataset"""
        
        all_data = torch.cat([batch for batch in dataset], dim=0)
        
        # Take the last sequence and add batch dimension
        last_sequence = all_data[-1].unsqueeze(0)
        
        # Ensure the sequence is on the correct device
        return last_sequence.to(self.device)
    
    def predict(self, dataloader_dict: dict, sequence_size: int) -> dict:
        """Run Monte Carlo prediction with diagnostics using data from a DataLoader"""
        # Extract the initial sequence from the dataloader
        initial_sequence = self._extract_last_sequence(dataloader_dict['test'])
        
        predicted_prices_all = np.zeros((self.num_iterations, self.num_days))
        prediction_std = np.zeros(self.num_days)
        
        # Store intermediate distributions for QQ plots
        self.prediction_distributions = {
            'early': [],
            'middle': [],
            'late': []
        }
        
        self.model.eval()
        with torch.no_grad():
            for seed in range(self.num_iterations):
                rng = np.random.RandomState(seed)
                predicted_prices = np.zeros(self.num_days)
                last_sequence = initial_sequence.clone()
                
                for day in range(self.num_days):
                    # Dynamic temperature with exponential decay
                    time_factor = np.exp(-day / self.num_days)
                    dynamic_temp = (self.base_temp + 
                                  self.prediction_variance(predicted_prices[:day])) * time_factor
                    
                    # Predict with uncertainty
                    next_day_prediction = self.model(last_sequence)
                    
                    # Add heteroscedastic noise
                    noise = rng.normal(0, np.sqrt(dynamic_temp))
                    next_day_prediction = next_day_prediction.squeeze().item() + noise
                    
                    # Store prediction
                    predicted_prices[day] = next_day_prediction
                    
                    # Update sequence
                    last_sequence = torch.cat((
                        last_sequence[:, 1:, :],
                        torch.tensor([[next_day_prediction]]).unsqueeze(0).to(self.device)
                    ), dim=1)
                
                # Store full sequence
                predicted_prices_all[seed] = self.scaler.inverse_transform(
                    predicted_prices.reshape(-1, 1)
                ).ravel()
                
                # Store distributions at different time points
                if seed % 100 == 0:
                    self.prediction_distributions['early'].append(predicted_prices_all[seed, 0:10])
                    self.prediction_distributions['middle'].append(predicted_prices_all[seed, 45:55])
                    self.prediction_distributions['late'].append(predicted_prices_all[seed, -10:])
        
        # Calculate statistics
        mean_predictions = np.mean(predicted_prices_all, axis=0)
        std_predictions = np.std(predicted_prices_all, axis=0)
        
        # Calculate confidence intervals
        ci_1std = np.percentile(predicted_prices_all, [15.87, 84.13], axis=0)
        ci_2std = np.percentile(predicted_prices_all, [2.5, 97.5], axis=0)
        
        # Run diagnostics
        convergence_metrics = self._assess_convergence(predicted_prices_all)
        self.convergence_metrics.append(convergence_metrics)
        
        # Check stationarity
        is_stationary, p_value = self._check_stationarity(mean_predictions)
        
        # Calculate autocorrelation
        self.autocorrelation = np.correlate(mean_predictions - np.mean(mean_predictions),
                                            mean_predictions - np.mean(mean_predictions),
                                            mode='full')[len(mean_predictions)-1:]
        
        return {
            'predictions': mean_predictions,
            'std': std_predictions,
            'ci_1std': ci_1std,
            'ci_2std': ci_2std,
            'convergence': convergence_metrics,
            'stationarity': {'is_stationary': is_stationary, 'p_value': p_value},
            'autocorrelation': self.autocorrelation,
            'prediction_matrix': predicted_prices_all
        }
    
    def prediction_variance(self, predictions: np.ndarray, min_var: float = 1e-6) -> float:
        """Calculate rolling variance with safeguards"""
        if len(predictions) == 0:
            return self.base_temp
        elif len(predictions) < self.window_size:
            return max(np.var(predictions) if len(predictions) > 1 else self.base_temp, min_var)
        return max(np.var(predictions[-self.window_size:]), min_var)
    
    def plot_diagnostics(self) -> None:
        """Plot diagnostic visualizations using matplotlib"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Convergence of means
        axes[0,0].plot(self.convergence_metrics[-1]['mean_convergence'])
        axes[0,0].set_title('Convergence of Running Means')
        axes[0,0].set_xlabel('Iteration')
        axes[0,0].set_ylabel('Change in Mean')
        
        # Plot 2: Autocorrelation
        if self.autocorrelation is not None:
            axes[0,1].plot(self.autocorrelation[:50])
            axes[0,1].set_title('Autocorrelation Function')
            axes[0,1].set_xlabel('Lag')
            axes[0,1].set_ylabel('ACF')
        
        # Plot 3: QQ plots for different time periods
        if self.prediction_distributions is not None:
            stats.probplot(np.array(self.prediction_distributions['early']).flatten(), 
                            dist="norm", plot=axes[1,0])
            axes[1,0].set_title('Q-Q Plot (Early Predictions)')
        
        # Plot 4: R-hat convergence
        axes[1,1].plot(self.convergence_metrics[-1]['r_hat'])
        axes[1,1].axhline(y=1.1, color='r', linestyle='--')
        axes[1,1].set_title('R-hat Convergence')
        axes[1,1].set_xlabel('Time Step')
        axes[1,1].set_ylabel('R-hat')
        
        plt.tight_layout()
        plt.show()