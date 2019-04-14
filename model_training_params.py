param_dict = {
    # ------------------
    # GENERAL PARAMETERS
    # ------------------
    'dscovr_path' : '',
    # ---------------------------
    # DATA PREPARATION PARAMETERS
    # ---------------------------
    # Variable to be predicted. OPTIONS:
    #   - ['RX', 'RY']      Root-mean-square dX/dt and dY/dt values
    #   - ['RGIC']          Root-mean-square GIC values
    'pred_vars' : ['RGIC'],
    # Determine if input includes standard deviation sigma (bool, if True number of input features increases from 7 to 10):
    'use_sigma' : False,
    # Interval over which mean is taken in minutes (10 for dB/dt predictions, 30 for GIC):
    'interval' : 30,
    # Data offset between input and output in minutes (default=0, this needs testing):
    'tau' : 0,
    # Number of timesteps in each sample (default=300):
    'sample_timesteps' : 200,
    # Overlap in data points of time series samples:
    'sample_overlap' : 10,
    # Shuffles the samples (bool):
    'shuffle_samples' : True,
    # ------------------------------
    # LSTM MODEL TRAINING PARAMETERS
    # ------------------------------
    # Batch size (if None, all samples are used):
    'batch_size' : 1,
    # LSTM iterations
    'epochs' : 400,
    # Number of hidden nodes in LSTM:
    'hidden_nodes' : 5,
    # Number of LSTM layers:
    'lstm_layers' : 2,
    # Use stateful LSTM:
    'stateful' : False
    }