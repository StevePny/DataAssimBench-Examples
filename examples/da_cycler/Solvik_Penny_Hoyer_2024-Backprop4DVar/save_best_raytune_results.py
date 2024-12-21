"""
Simple script for loading best raytune results and saving to
a consolidated csv.
"""

import pandas as pd
import numpy as np

raytune_results_dict = {
    'l96_system_dim': './out_rev/l96/raytune_l96_v4.csv',
    'l96_obs_error': './out_rev/l96/raytune_l96_heatmap_v4.csv',
    'pyqg': './out_rev/pyqg/pyqg_jax_raytune_v2.csv',
    'qgs': './out_rev/qgs/raytune_qgs_v3_hessian.csv'
}

# System dim experiments
raytune_system_dim_results = pd.read_csv(raytune_results_dict['l96_system_dim'])
raytune_system_dim_results['trialnum'] = raytune_system_dim_results.index
raytune_system_dim_results.index = np.arange(raytune_system_dim_results.shape[0])
rows_to_get = raytune_system_dim_results.groupby(['system_dim']).idxmin(numeric_only=True)['rmse']
best_results_system_dim = raytune_system_dim_results.loc[rows_to_get]
best_results_system_dim = best_results_system_dim[['system_dim', 'config/lr', 'config/lr_decay']]
best_results_system_dim['obs_sd'] = 0.5
best_results_system_dim['num_obs'] = (best_results_system_dim['system_dim']/2).astype(int)
best_results_system_dim.rename(columns={'config/lr': 'learning_rate',
                                        'config/lr_decay': 'lr_decay'},
                               inplace=True)
best_results_system_dim['experiment'] = 'l96_system_dim'

# Obs error experiments
raytune_heatmap_results = pd.read_csv(raytune_results_dict['l96_obs_error'])
raytune_heatmap_results['trialnum'] = raytune_heatmap_results.index
raytune_heatmap_results.index = np.arange(raytune_heatmap_results.shape[0])
rows_to_get = raytune_heatmap_results.groupby(['num_obs','obs_sd']).idxmin(numeric_only=True)['rmse']
best_results_heatmap = raytune_heatmap_results.loc[rows_to_get]
best_results_heatmap = best_results_heatmap[['system_dim', 'config/lr', 'config/lr_decay',
                                                   'obs_sd', 'num_obs']]
best_results_heatmap.rename(columns={'config/lr': 'learning_rate',
                                        'config/lr_decay': 'lr_decay'},
                               inplace=True)
best_results_heatmap['experiment'] = 'l96_obs_error'

# PyQG
raytune_pyqg_results = pd.read_csv(raytune_results_dict['pyqg'])
raytune_pyqg_results['trialnum'] = raytune_pyqg_results.index
raytune_pyqg_results.index = np.arange(raytune_pyqg_results.shape[0])
rows_to_get = raytune_pyqg_results.groupby(['system_dim_xy']).idxmin(numeric_only=True)['rmse']
best_results_pyqg = raytune_pyqg_results.loc[rows_to_get]
best_results_pyqg['system_dim'] = 2 * best_results_pyqg['system_dim_xy']**2
best_results_pyqg = best_results_pyqg[['system_dim', 'config/lr', 'config/lr_decay']]
best_results_pyqg.rename(columns={'config/lr': 'learning_rate',
                                        'config/lr_decay': 'lr_decay'},
                               inplace=True)
best_results_pyqg['obs_sd'] = np.nan
best_results_pyqg['num_obs'] = (best_results_pyqg['system_dim']/2).astype(int)
best_results_pyqg['experiment'] = 'pyqg'

# QGS
raytune_qgs_results = pd.read_csv(raytune_results_dict['qgs'])
best_row = raytune_qgs_results['rmse'].idxmin()
best_results_qgs = raytune_qgs_results.loc[[best_row]]
best_results_qgs = best_results_qgs[['config/lr', 'config/lr_decay']]
best_results_qgs['system_dim'] = 20
best_results_qgs['obs_sd'] = np.nan
best_results_qgs['num_obs'] = 10
best_results_qgs.rename(columns={'config/lr': 'learning_rate',
                                        'config/lr_decay': 'lr_decay'},
                               inplace=True)
best_results_qgs['experiment'] = 'qgs'


# Combine
full_results_df = pd.concat(
    [best_results_system_dim, best_results_heatmap, best_results_pyqg, best_results_qgs]
)


# Reorder
full_results_df = full_results_df[['experiment', 'system_dim', 'num_obs', 'obs_sd', 'learning_rate', 'lr_decay']]
full_results_df.to_csv('./out_rev/raytune_best_results.csv', index=False)