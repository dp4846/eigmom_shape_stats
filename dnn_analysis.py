import numpy as np
import matplotlib.pyplot as plt
import src
import xarray as xr
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from joblib import Parallel, delayed
import multiprocessing

def run_sim(X, Y, Ms, num_moments, alpha, num_bootstraps, remove_constant, bias_frac, seed, shuffle=False):
    #np.random.seed(seed)
    M, D = Y.shape
    res = []
    for m in (Ms):
        if shuffle:
            inds_m1 = np.random.choice(M, m, replace=False)
            inds_m2 = np.random.choice(M, m, replace=False)
            X_sub = X[inds_m1, :]
            Y_sub = Y[inds_m2, :]
        else:
            inds_m = np.random.choice(M, m, replace=False)
            X_sub = X[inds_m, :]
            Y_sub = Y[inds_m, :]
        est, var, bias, naive_est = src.full_similarity_metric_estimator(X_sub, Y_sub, num_moments=num_moments, alpha=alpha, 
                                                    num_bootstraps=num_bootstraps, remove_constant=remove_constant,
                                                                    bias_frac=bias_frac)
        res.append([est, var, bias, naive_est])
    return res
np.random.seed(2)
num_cpus = multiprocessing.cpu_count()
#change to where ever you have data.
raw_data_dir = './data/'
Y = np.load(raw_data_dir + 'model2_activations.npy')
M, D = Y.shape
neur_inds = np.random.choice(D, 50, replace=False)
stim_inds = np.random.choice(M, 200000, replace=False)
X = np.load(raw_data_dir + 'model1_activations.npy')[stim_inds][:, neur_inds]
Y = np.load(raw_data_dir + 'model2_activations.npy')[stim_inds][:, neur_inds]
X = X - X.mean(0)
Y = Y - Y.mean(0)
X = X/X.std(0)
Y = Y/Y.std(0)
M, D = Y.shape
cov_est = (X.T @ Y)/(M)#TODO extend to more than two repeats
svd = TruncatedSVD(n_components=1, n_iter=7, random_state=42)
svd.fit(cov_est)
s1_est = svd.singular_values_[0]
X /= s1_est ** 0.5
Y /= s1_est ** 0.5
denom_est = (np.trace(X.T@X/M)*np.trace(Y.T@Y/M))**0.5 
#ground truth
cov_est = (X.T @ Y)/(M)
naive_est_nuc_norm  = np.linalg.norm(cov_est, ord='nuc')
ground_truth = naive_est_nuc_norm/denom_est
print(ground_truth)
#%%
num_moments = 7
alpha = 1
num_bootstraps = 50
remove_constant = True
bias_frac = 1
Ms = np.arange(50, 500, 50)
n_sims = 500
seeds = np.arange(n_sims)
num_cpus = multiprocessing.cpu_count()
res = Parallel(n_jobs=num_cpus)(delayed(run_sim)(X, Y, Ms, num_moments, alpha, num_bootstraps, remove_constant, bias_frac, seed) for seed in seeds)
res = np.array(res)
#%%
plt.figure(figsize=(3,2))
eig_est_mu = res.mean(0)[:,0]
eig_est_sd = res.std(0)[:,0]
naive_est_mu = res.mean(0)[:,-1]
naive_est_sd = res.std(0)[:,-1]

plt.errorbar(Ms, eig_est_mu, yerr=2*eig_est_sd/n_sims**0.5, label='Moment est.')
plt.errorbar(Ms, naive_est_mu, yerr=2*naive_est_sd/n_sims**0.5, label='Plug-in est.')
plt.plot(Ms, [ground_truth,]*len(Ms), label='ground truth', c='k')

plt.legend(title=r'$\mu \pm$ SE', loc=(1.05,0))
plt.ylim(0, 0.5)
plt.xlabel('# stimuli (M)')
plt.ylabel('Similarity')
plt.title('Dimensionality=' + str(D*2))
plt.savefig('bias_M_rel.pdf', bbox_inches='tight')
#%%
#raw responses which we will sub sample
Y = np.load(raw_data_dir + 'model2_activations.npy')
M, D = Y.shape
neur_inds = np.random.choice(2048, 2048, replace=False)
stim_inds = np.random.choice(432064, 432064, replace=False)
X_all = np.load(raw_data_dir + 'model1_activations.npy')[stim_inds][:, neur_inds]
Y_all = np.load(raw_data_dir + 'model2_activations.npy')[stim_inds][:, neur_inds]
X_all = X_all - X_all.mean(0)
Y_all = Y_all - Y_all.mean(0)

#%%
num_moments = 7
alpha = 1
num_bootstraps = 50
remove_constant = True
bias_frac = 1
Ms = [100,]
n_sims = 50

results = []
for i in tqdm(range(1,501)):
    #select some random neurons
    neur_inds = np.random.choice(D, 50, replace=False)
    X = X_all[:, neur_inds]
    Y = Y_all[:, neur_inds]

    cov_est = (X.T @ Y)/(M)

    svd = TruncatedSVD(n_components=1, n_iter=7, random_state=42)
    svd.fit(cov_est)
    s1_est = svd.singular_values_[0]
    X /= s1_est**0.5
    Y /= s1_est**0.5
    # rescale them
    cov_x = X.T@X/M
    cov_y = Y.T@Y/M
    denom_est = (np.trace(cov_x)*np.trace(cov_y))**0.5 
    
    cov_est = (X.T @ Y)/(M)
    naive_est_nuc_norm  = np.linalg.norm(cov_est, ord='nuc')
    ground_truth = naive_est_nuc_norm/denom_est#get ground truth

    #calculate effective dimensionality.
    edim_x = np.trace(cov_x)**2/np.trace(cov_x@cov_x)
    edim_y = np.trace(cov_y)**2/np.trace(cov_y@cov_y)
    e_dim = (edim_x*edim_y)**0.5

    #run 50 simulations across stimuli get mean and variance of plug
    seeds = np.arange(1, 1 + n_sims)*i
    num_cpus = multiprocessing.cpu_count()
    res = Parallel(n_jobs=num_cpus)(delayed(run_sim)(X, Y, Ms, num_moments, alpha, num_bootstraps, remove_constant, bias_frac, seed) for seed in seeds)
    res = np.array(res)
    eig_mu, plug_mu = res.mean(0).squeeze()[[0, -1]]
    eig_sd, plug_sd = res.mean(0).squeeze()[[0, -1]]
    results.append([eig_mu, plug_mu, eig_sd, plug_sd, e_dim, ground_truth])

plt.figure(figsize=(3,3))
res = np.array(results)
bias_plug_in = res[:, 1] - res[:, -1]
ed = res[:, -2]
plt.scatter(ed, bias_plug_in, s=1, c='k')
plt.xlabel('Effective dimensionality')
plt.ylabel('Plug-in estimator bias')
plt.ylim(0,0.3)
plt.title('r=' + str(np.corrcoef(ed, bias_plug_in)[0,1].round(2)))
plt.tight_layout()
plt.savefig('bias_dim_rel.pdf', bbox_inches='tight')