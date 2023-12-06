#%%
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import src
np.random.seed(2)
def plugin_prediction(m, n):
    return (n ** 1.5) * (8 / (3 * np.pi)) * (1 / np.sqrt(m))
 # %% FIGURE 3
n_sims = 100
M = 50
N = 50
pl_slope=0.
true_shape_metric = 0.0
num_moments = 7
naive_ests = []
mom_ests = []
Ms = np.arange(50, 1000, 50)

for M in tqdm(Ms):
    X, Y, singular_values, lam_x, lam_y, cov = src.sim_data(N, N, true_shape_metric, pl_slope, M, n_sims)
    singular_values = np.linalg.svd(cov[:N, N:])[1]
    true_eigmoms =[np.sum(singular_values**(p*2)) for p in range(0, num_moments+1)]
    true_denom = (np.sum(lam_x)*np.sum(lam_y))**0.5
    sample_cov = np.matmul(np.transpose(X, (0, 2, 1)) , Y)
    naive_est  = np.linalg.norm((sample_cov) / M, ord='nuc', axis=(1,2))
    naive_ests.append(naive_est)

    est_eigmoms = np.zeros((n_sims, num_moments+1))
    for i in tqdm(range(n_sims)): 
        est_eigmoms[i] = src.estimate_eigenmoments(X[i], Y[i], num_moments)
    eig_mom_cov = np.cov(est_eigmoms.T)

    gamma, bias, M_xt, xt = src.calc_gammas(
            X[0], Y[0], num_moments,
            eig_mom_cov=eig_mom_cov,
            max_eig = np.max(lam_x)**0.5, remove_constant=True, 
            return_poly_basis=True,
            bias_frac=1)
    mom_est =  est_eigmoms @ gamma
    mom_ests.append(mom_est)

#%% now plot naive est and plug in prediction
naive_ests_M = np.array(naive_ests)
mom_ests_M = np.array(mom_ests)

# %%
M = 200
naive_ests_N = []
mom_ests_N = []
Ns = np.arange(10, 100, 10)

for N in tqdm(Ns):
    X, Y, singular_values, lam_x, lam_y, cov = src.sim_data(N, N, true_shape_metric, pl_slope, M, n_sims)
    singular_values = np.linalg.svd(cov[:N, N:])[1]
    true_eigmoms =[np.sum(singular_values**(p*2)) for p in range(0, num_moments+1)]
    true_denom = (np.sum(lam_x)*np.sum(lam_y))**0.5
    sample_cov = np.matmul(np.transpose(X, (0, 2, 1)) , Y)
    naive_est  = np.linalg.norm((sample_cov) / M, ord='nuc', axis=(1,2))
    naive_ests_N.append(naive_est)

    est_eigmoms = np.zeros((n_sims, num_moments+1))
    for i in tqdm(range(n_sims)): 
        est_eigmoms[i] = src.estimate_eigenmoments(X[i], Y[i], num_moments)
    eig_mom_cov = np.cov(est_eigmoms.T)

    gamma, bias, M_xt, xt = src.calc_gammas(
            X[0], Y[0], num_moments,
            eig_mom_cov=eig_mom_cov,
            max_eig = np.max(lam_x)**0.5, remove_constant=True, 
            return_poly_basis=True,
            bias_frac=1)
    mom_est =  est_eigmoms @ gamma
    mom_ests_N.append(mom_est)
#%%
naive_ests_N = np.array(naive_ests_N)
mom_ests_N = np.array(mom_ests_N)
#%%
plt.figure(figsize=(4,2))
ylim=(-4,50)
plt.subplot(1,2,1)
plt.plot(Ms, plugin_prediction(Ms, 50), label='plug-in theory', c='g')
plt.errorbar(Ms, y=naive_ests_M.mean(1), yerr=naive_ests_M.std(1),  label='plug-in est', c='blue')
plt.errorbar(Ms, y=mom_ests_M.mean(1), yerr=mom_ests_M.std(1),  label='moment est', c='orange')
plt.plot([0,1000], [0,0], label='ground truth', ls='--', c='k', zorder=1000)
plt.xlabel('# sampled inputs (M)')
plt.ylabel(r'$||\Sigma_{i,j}||^*$', rotation=0, labelpad=24)
handles, labels = plt.gca().get_legend_handles_labels()
order = [1, 3, 2, 0]  # Specify the legend order
plt.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=8)

plt.ylim(ylim)
plt.title('N=50')
plt.subplot(1,2,2)

plt.plot([0,100], [0,0], label='ground truth', ls='--', c='k', zorder=1000)
plt.plot(Ns, plugin_prediction(200, Ns), c='green')
plt.errorbar(Ns, y=naive_ests_N.mean(1), yerr=naive_ests_N.std(1), c='blue')
plt.errorbar(Ns, y=mom_ests_N.mean(1), yerr=mom_ests_N.std(1), c='orange')


plt.xlabel('# neurons (N)')
plt.title('M=200')
plt.gca().set_yticklabels([])
plt.ylim(ylim)
plt.savefig('plug_in_bias_theory.pdf', bbox_inches='tight')
# %%
