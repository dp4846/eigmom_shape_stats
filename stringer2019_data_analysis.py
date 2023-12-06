#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
import src
np.random.seed(2)
#change to wherever you have the data
raw_data_dir = './data/'
fns = [fn for fn in os.listdir(raw_data_dir) if 'natimg2800_' in fn ]
n_rec = len(fns)
# for holding onto eigenspectra raw results
fn_nms = [fns[rec].split('/')[-1].split('.')[0] for rec in range(n_rec)]
rec = fn_nms[0]
sub_samp = 1
das = []
for rec in fn_nms:
    da = xr.open_dataset(raw_data_dir + rec + '.nc')['resp'][..., ::sub_samp]
    das.append(da/1e4)#scaling to prevent explosion of eigenmoments

# %% FIG 4 data
N = 40
M = 400
max_chunks = np.max([das[i].shape[1]//(M*2) for i in range(7)])
n_recs = len(das)
# 'zero' condition means same neurons different stimuli so similarity is 0
# 'one' condition means same neurons same stimuli so similarity is 1
# 'data' condition means different neurons same stimuli so similarity is unknown                            
sim_types = ['zero', 'one', 'data']
n_sim = max_chunks
da_res = xr.DataArray(np.zeros((n_recs, 3, n_sim, 4)), dims=['rec', 'sim_type', 'sim', 'metric',],
                        coords={'rec':range(n_recs), 'sim_type':sim_types, 'sim':range(n_sim),
                                 'metric': ['est', 'var', 'bias', 'naive_est'],
                                })
da_res[...] = np.nan
for i in range(len(das)):
    da = das[i]
    n_rep, n_stim, n_neur = das[i].shape
    R = da.values
    R = R - R.mean((0,1), keepdims=True)#subtract off average across stimuli
    #chunk indices of stimuli into chunks of size M
    rand_inds = np.random.choice(n_stim, size=n_stim, replace=False)
    chunk_inds = [rand_inds[np.arange(j*M*2, (j+1)*M*2)] for j in range(n_stim//(M*2))]
    n_sim = len(chunk_inds)
    #get the snr of each neuron
    noise_var = R.var(0).mean(0)
    sig_var = R.mean(0).var(0)
    snr = sig_var/noise_var
    snr[np.isnan(snr)] = 0#noise var will sometimes be 0 giving nans
    #sort neurons by snr
    neur_ind = np.random.choice(n_neur, size=N*2, replace=False)
    neur_ind = np.argsort(snr)[::-1][:N*2]
    neur_ind_x = neur_ind[1::2]
    neur_ind_y = neur_ind[::2]

    for sim_type in sim_types:
        for sim in range(n_sim):
            stim_ind = chunk_inds[sim]
            if sim_type == 'zero':#same neurons different stimuli
                stim_ind_x = stim_ind[:M]
                stim_ind_y = stim_ind[M:]
                X = R[:, stim_ind_x][..., neur_ind_x]
                Y = R[:, stim_ind_y][..., neur_ind_y]
            elif sim_type == 'one':#same neurons same stimuli
                stim_ind_x = stim_ind[:M]
                stim_ind_y = stim_ind[M:]
                X = R[:, stim_ind_x][..., neur_ind_x]
                Y = R[:, stim_ind_x][..., neur_ind_x]
            elif sim_type == 'data':#different neurons same stimuli
                stim_ind_x = stim_ind[:M]
                stim_ind_y = stim_ind[M:]
                X = R[:, stim_ind_x][..., neur_ind_x]
                Y = R[:, stim_ind_x][..., neur_ind_y]

            est, var, bias, naive_est = src.full_signal_similarity_metric_estimator(X, Y, num_moments=10, alpha=1., 
                                                        num_bootstraps=50, remove_constant=True,
                                                                        bias_frac=1.)
            da_res.loc[i, sim_type, sim] = np.array([est, var, bias, naive_est])
#%% now estimate shape metric across all stimuli
full_res= []
for i in (range(7)):
    da = das[i]
    n_rep, n_stim, n_neur = das[i].shape

    R = da.values
    R = R - R.mean((0,1), keepdims=True)#subtract off average across stimuli

    #get the snr of each neuron
    noise_var = R.var(0).mean(0)
    sig_var = R.mean(0).var(0)
    snr = sig_var/(noise_var)
    snr[np.isnan(snr)] = 0
    #sort neurons by snr so just estimating similarity across top N neurons
    neur_ind = np.argsort(snr)[::-1][:N*2]
    neur_ind_x = neur_ind[::2]
    neur_ind_y = neur_ind[1::2]
    X = R[:, :][..., neur_ind_x]
    Y = R[:, :][..., neur_ind_y]
    est, var, bias, naive_est = src.full_signal_similarity_metric_estimator(X, Y, num_moments=10, alpha=1., 
                                            num_bootstraps=100, remove_constant=True,
                                                            bias_frac=1.)
    full_res.append([est, var, bias, naive_est ])
res = np.array(full_res)
#%% FIGURE 4 plots
s = 0.7
last_sim = 0
fig, ax = plt.subplots(1,4, figsize=(12*s, 4*s), dpi=400)
for rec in range(7):
    a_da_res = da_res[rec].dropna('sim')
    n_sim = len(a_da_res.coords['sim'])
    for i, sim_type in enumerate(sim_types):
        if sim_type == 'zero':
            ground_truth = 0.
            ax[i].plot(np.arange(n_sim)+last_sim, np.ones(n_sim)*ground_truth, color='black', label='Ground truth', ls='--', zorder=10)
        elif sim_type == 'one':
            ground_truth = 1.
            ax[i].plot(np.arange(n_sim)+last_sim, np.ones(n_sim), color='black', label='Ground truth', ls='--', zorder=10)
        elif sim_type == 'data':
            ground_truth = None
        bias = a_da_res.loc[sim_type, :,'bias'].values
        sd = (a_da_res.loc[sim_type, :, 'var']**0.5).values
        ci = sd*1.96 + bias
        ax[i].plot(np.arange(n_sim)+last_sim, a_da_res.loc[sim_type, :, 'est'], color='orange', label='Moment est.')
        ax[i].fill_between(np.arange(n_sim)+last_sim, a_da_res.loc[sim_type, :, 'est']-bias, a_da_res.loc[sim_type, :, 'est']+bias, color='orange', alpha=0.5,
        label='Bias')
        ax[i].fill_between(np.arange(n_sim)+last_sim, a_da_res.loc[sim_type, :, 'est']-ci, a_da_res.loc[sim_type, :, 'est']+ci, 
                            color='orange', alpha=0.2, label='CI')
        ax[i].plot(np.arange(n_sim)+last_sim, a_da_res.loc[sim_type, :, 'naive_est'], color='blue', label='Naive est.')
        ax[i].set_ylim([-0.5, 1.5])
        ax[i].set_xticks([1, 4, 7, 9.5, 12, 15,18])
        
        if i == 0:
            ax[i].set_ylabel('Estimated\n similarity \n'r'$ \left(\frac{{\widehat{||\Sigma_{1,2}||}}_{*}}{\sqrt{tr[\widehat{\Sigma_{1}}]tr[\widehat{\Sigma_{2}}}]}\right)$', 
                    rotation=0, labelpad=50, fontsize=12, ha='center', va='center')
            ax[i].set_xlabel('Neural recording')
            ax[i].set_xticklabels(range(1,8))
        else:
            ax[i].set_yticklabels([])
            ax[i].set_xticklabels([])
        if rec==0 and i==0:
            ax[i].legend(['Ground truth', 'Moment est.', 'Bias', 'CI', 'Plug-in est.' ], loc=(-1.2,-0.2), fontsize=8)
    last_sim += n_sim

est = res[:,0]
bias = res[:,2]
sd = res[:,1]**0.5
ci = sd*1.96 + bias
ax[3].plot(np.arange(1,8),est, color='orange', label='Moment est.')

ax[3].fill_between(np.arange(1,8), est-bias, est+bias, color='orange', alpha=0.5,
label='Bias')
ax[3].fill_between(np.arange(1,8), est-ci, est+ci, 
                            color='orange', alpha=0.2, label='CI')
ax[3].plot(np.arange(1,8), res[:,-1], label='Plug-in est.', color='blue')
ax[3].set_ylim(-0.5,1.5)
i=3
#remove tick labels
ax[i].set_yticklabels([])
ax[i].set_xticklabels([])
ax[i].set_xticks(np.arange(1,8))
titles = ['Ground truth=0', 'Ground truth=1', 'Raw data', 'Raw data all stimuli']
for i in range(4):
    ax[i].grid(axis='y')
    ax[i].set_title(titles[i])
plt.savefig('./neural_dat_sim.pdf', bbox_inches='tight')