#!/usr/bin/env python\n"

from .gic import prepare_B_for_E_calc, calc_E_using_plane_wave_method
from .gic import calc_ab_for_GIC_from_E

from .sampling import sample_quiet_and_active_times, high_val_decay, calc_t_pers_back
from .sampling import SolarWindFeatureEncoder, extract_datasets_by_years
from .sampling import extract_feature_samples, create_target_array, create_persistence_array

from .training import BasicAttention, DataStorer, min_max_loss
from .training import get_model, fit_model

from .tools import extract_time_from_pos, extract_local_time_variables
from .tools import calc_event_rates, compute_CM, compute_metrics, plot_on_time_series