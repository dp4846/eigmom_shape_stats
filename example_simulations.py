#%%
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import src
np.random.seed(2)
 # %% FIGURE 3
n_sims = 10000
M = 50
N = 50
pl_slope=0.
true_shape_metric = 0.4
gammas = []
num_moments = 25
all_ests = []
lams = []
N_x = N
N_y = N
X, Y, singular_values, lam_x, lam_y, cov = src.sim_data(N_x, N_y, true_shape_metric, pl_slope, M, n_sims)
singular_values = np.linalg.svd(cov[:N, N:])[1]
true_eigmoms =[np.sum(singular_values**(p*2)) for p in range(0, num_moments+1)]
true_denom = (np.sum(lam_x)*np.sum(lam_y))**0.5
est_eigmoms = np.zeros((n_sims, num_moments+1))
for i in tqdm(range(n_sims)): 
    est_eigmoms[i] = src.estimate_eigenmoments(X[i], Y[i], num_moments)
eig_mom_cov = np.cov(est_eigmoms.T)
sample_cov = np.matmul(np.transpose(X, (0, 2, 1)) , Y)
naive_est  = np.linalg.norm((sample_cov) / M, ord='nuc', axis=(1,2))/true_denom
res = []
bias_fracs = np.logspace(-2, 0, 25)
gammas =[]
for bias_frac in tqdm(bias_fracs):
    gamma, bias, M_xt, xt = src.calc_gammas(
            X[0], Y[0], num_moments,
            alpha=0.5,
            eig_mom_cov=eig_mom_cov,
            max_eig = np.max(lam_x)**0.5, remove_constant=True, 
            return_poly_basis=True,
            bias_frac=bias_frac)
    gammas.append(gamma)
    var = gamma @ eig_mom_cov @ gamma / true_denom**2
    est =  est_eigmoms @ gamma / true_denom
    true_mean = true_eigmoms @ gamma / true_denom
    est_mean = np.mean(est)
    est_var = np.var(est)
    res.append([est_mean, est_var, bias/true_denom, var, true_mean])
res = np.array(res)
truth = np.array([true_shape_metric,]*len(bias_fracs))

#%% FIGURE 3 plotting
s = 0.8
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10*s, 5.5*s), dpi=300, sharey=False)
ax1.plot(bias_fracs, truth, c='k', ls='--', label='Truth')


ind = (res[:,1]**0.5/n_sims**0.5)<0.01
ax1.set_ylabel('Similarity \n' r'$ \left(\frac{{||\Sigma_{1,2}||}_{*}}{\sqrt{tr[\Sigma_{1}]tr[\Sigma_{2}}]}\right)$', rotation=0, labelpad=30, va='center')

ax1.errorbar(x=bias_fracs[ind], y=res[ind,0], yerr=res[ind,1]**0.5, label='Moment est. simulation', c='orange')
ax1.errorbar(bias_fracs, [np.mean(naive_est),]*len(bias_fracs), yerr=np.std(naive_est), label='Plug-in est. simulation')
ax1.fill_between(bias_fracs, true_shape_metric - bias_fracs, true_shape_metric + bias_fracs, alpha=0.2, label='User defined bias bound')
ax1.fill_between(bias_fracs, truth - res[:,2], truth + res[:,2], alpha=0.5, label='Moment est. bias bound', color='cyan')
ax1.invert_xaxis()
ax1.semilogx()
ax1.set_ylim(0,1)
ax1.set_xlim(1, 0.01)
ax1.set_xticks([1, 0.1, 0.01])
ax1.set_xticklabels(['1', '0.1', '0.01'])
ax1.set_xlabel('User defined bias bound')
ax1.plot(bias_fracs, res[:,-1], c='orange', ls=':', label='Moment est. expectation')

ax1.legend(loc=(-0.1,-0.8), borderaxespad=0., fontsize=6, ncol=2)

#now make approximation plot
lss = ['-', '--', '-.', ':']
#center y label
ax2.set_ylabel('Cross-covariance\nsingular\nvalues\nnormalized\n' r'$\left(s_i \frac{N}{\sqrt{tr[\Sigma_{1}]tr[\Sigma_{2}}]}\right)$', 
                rotation=0, labelpad=45, va='center',)
ax2.plot(xt, N*(xt**0.5)/true_denom, c='k', ls='--', label=r'$\sqrt{x}$',alpha=0.5)
#ax2.plot([np.mean(singular_values)**2,]*2, [0, 1], c='grey')
ax2.scatter(singular_values**2, N*singular_values/true_denom, c='k', label='Ground truth ' r'$s_i$', marker='o', s=20, zorder=10,alpha=0.5)

xt_sing = singular_values**2
M_sing = np.array([xt_sing**i for i in range(0, num_moments + 1)]).T
gammas = np.array(gammas)
labels = ['Poly. approx. high bias low variance', 'Poly. approx. low bias high variance'][::-1]
for i, gamma in enumerate(gammas[[0,-1]]):
    ax2.plot(xt, N*M_xt@gamma/true_denom, c='orange', label=labels[i], ls=lss[1:][i])
    #ax2.scatter(xt_sing, N*M_sing@gamma/true_denom, c='orange', label='Poly. approx. ' r'$s_i$', marker='.')
ax2.set_xlabel(r'$s_i^2$')
ax2.legend(loc=(0, -0.8), borderaxespad=0., fontsize=6)
ax2.set_ylim(0, 1.)
ax2.set_yticklabels([])
ax2.grid()
plt.tight_layout()
plt.savefig('./shape_metric_bias_bound_temp.pdf', bbox_inches='tight')
# %% FIGURE 2 DATA

#FIGURE 2A shape metrics fixing M and N
shape_metrics = np.linspace(0, 1, 20)
n_sims = 100
# vary true shape metric f
M_shape_metric_sim = 200
N_shape_metric_sim = 100
pl_slope=0.
num_moments = 15
res_shape_metrics = []
for true_shape_metric in tqdm(shape_metrics):
    X, Y, singular_values, lam_x, lam_y, cov = src.sim_data(N_shape_metric_sim, N_shape_metric_sim, true_shape_metric, pl_slope, M_shape_metric_sim, n_sims)
    true_denom = np.sqrt(np.sum(lam_x)*np.sum(lam_y))
    est_eigmoms = np.zeros((n_sims, num_moments+1))
    for i in (range(n_sims)): 
        est_eigmoms[i] = src.estimate_eigenmoments(X[i], Y[i], num_moments)
    eig_mom_cov = np.cov(est_eigmoms.T)
    sample_cov = np.matmul(np.transpose(X, (0, 2, 1)) , Y)
    naive_est  = np.linalg.norm((sample_cov) / M_shape_metric_sim, ord='nuc', axis=(1,2))/true_denom
    gamma, bias, M_xt, xt = src.calc_gammas(
            X[0], Y[0], num_moments,
            alpha=1.,
            eig_mom_cov=eig_mom_cov,
            max_eig = np.max(lam_x)**0.5, remove_constant=True, 
            return_poly_basis=True,
            bias_frac=1)
    est =  est_eigmoms @ gamma / true_denom
    res_shape_metrics.append([naive_est, est])
res_shape_metrics = np.array(res_shape_metrics)

# FIGURE 2B now vary M while fixing N and shape metric
n_sims = 1000
Ms = [300, 400, 500, 600, 700, 800, 1000]
N_Msim = 100
true_shape_metric = 0.2
res_M = []
for m in tqdm(Ms):
    X, Y, singular_values, lam_x, lam_y, cov = src.sim_data(N_Msim, N_Msim, true_shape_metric, pl_slope, m, n_sims)
    true_denom = (np.sum(lam_x)*np.sum(lam_y))**0.5
    est_eigmoms = np.zeros((n_sims, num_moments+1))
    for i in (range(n_sims)): 
        est_eigmoms[i] = src.estimate_eigenmoments(X[i], Y[i], num_moments)
    eig_mom_cov = np.cov(est_eigmoms.T)
    sample_cov = np.matmul(np.transpose(X, (0, 2, 1)) , Y)
    naive_est  = np.linalg.norm((sample_cov) / m, ord='nuc', axis=(1,2))/true_denom
    gamma, bias, M_xt, xt = src.calc_gammas(
            X[0], Y[0], num_moments,
            alpha=1.0,
            eig_mom_cov=eig_mom_cov,
            max_eig = np.max(lam_x)**0.5, remove_constant=True, 
            return_poly_basis=True,
            bias_frac=0.05)
    est =  est_eigmoms @ gamma / true_denom
    res_M.append([naive_est, est])

# FIGURE 2C now vary N while fixing M and shape metric
Ns = [10, 25, 50, 75, 100, 125, 150, 175, 200]
M_Nsim = 100
res_N = []
for n in tqdm(Ns):
    X, Y, singular_values, lam_x, lam_y, cov = src.sim_data(n, n, true_shape_metric, pl_slope, M_Nsim, n_sims)
    true_denom = (np.sum(lam_x)*np.sum(lam_y))**0.5
    est_eigmoms = np.zeros((n_sims, num_moments+1))
    for i in (range(n_sims)): 
        est_eigmoms[i] = src.estimate_eigenmoments(X[i], Y[i], num_moments)
    eig_mom_cov = np.cov(est_eigmoms.T)
    sample_cov = np.matmul(np.transpose(X, (0, 2, 1)) , Y)
    naive_est  = np.linalg.norm((sample_cov) / M_Nsim, ord='nuc', axis=(1,2))/true_denom
    gamma, bias, M_xt, xt = src.calc_gammas(
            X[0], Y[0], num_moments,
            alpha=1.,
            eig_mom_cov=eig_mom_cov,
            max_eig = np.max(lam_x)**0.5, remove_constant=True, 
            return_poly_basis=True,
            bias_frac=0.1)
    est =  est_eigmoms @ gamma / true_denom
    res_N.append([naive_est, est])
res_N = np.array(res_N)

# FIGURE 2D now make a simulation for CIs
M_cisim = 200
N_cisim = 100
n_sims = 100
true_shape_metric = 0.2
X, Y, singular_values, lam_x, lam_y, cov = src.sim_data(N_cisim, N_cisim, true_shape_metric, pl_slope, M_cisim, n_sims)
true_denom = (np.sum(lam_x)*np.sum(lam_y))**0.5
est_eigmoms = np.zeros((n_sims, num_moments+1))
for i in tqdm(range(n_sims)):
    est_eigmoms[i] = src.estimate_eigenmoments(X[i], Y[i], num_moments)
eig_mom_cov = np.cov(est_eigmoms.T)
sample_cov = np.matmul(np.transpose(X, (0, 2, 1)) , Y)
naive_est  = np.linalg.norm((sample_cov) / M_cisim, ord='nuc', axis=(1,2))/true_denom
gamma, bias, M_xt, xt = src.calc_gammas(
        X[0], Y[0], num_moments,        
        alpha=1.,
        eig_mom_cov=eig_mom_cov,
        max_eig = np.max(lam_x)**0.5, remove_constant=True,
        return_poly_basis=True,
        bias_frac=0.1)
est =  est_eigmoms @ gamma / true_denom
bias = bias/true_denom
var = gamma @ eig_mom_cov @ gamma / true_denom**2
ci_95 = 1.96*(var**0.5)
ci_var_bias = ci_95 + bias
res_ci = np.array([naive_est, est]).T


#%% FIGURE 2 plotting
s = 1.4
fig, ax = plt.subplots(1,4, figsize=(8*s,2*s))
y_ticks=[0,0.2,0.4,0.6,0.8,1]
ax[0].errorbar(shape_metrics, res_shape_metrics[:,1].mean(axis=1), yerr=res_shape_metrics[:,1].std(axis=1), label='Moment est. simulation', c='orange')
ax[0].errorbar(shape_metrics, res_shape_metrics[:,0].mean(axis=1), yerr=res_shape_metrics[:,0].std(axis=1), label='Plug-in est. simulation')
ax[0].plot(shape_metrics, shape_metrics, label='True', c='k', ls='--')
ax[0].set_xlabel('True similarity 'r'$ \left(\frac{{||\Sigma_{1,2}||}_{*}}{\sqrt{tr[\Sigma_{1}]tr[\Sigma_{2}}]}\right)$' )
ax[0].set_ylabel('Estimated similarity 'r'$ \left(\frac{{\hat{||\Sigma_{1,2}||}}_{*}}{\sqrt{tr[\Sigma_{1}]tr[\Sigma_{2}}]}\right)$' )
ax[0].set_aspect('equal', 'box')
ax[0].set_xlim(-0.1,1.1)
ax[0].set_ylim(-0.1,1.1)
#ticks should be the same
ax[0].set_xticks(y_ticks)
ax[0].set_yticks([0,0.2,0.4,0.6,0.8,1])
ax[0].set_title(' Estimator comparison \n M=' + str(M_shape_metric_sim) + ', N=' + str(N_shape_metric_sim))

ax[1].errorbar(Ms, np.array(res_M)[:,1].mean(axis=1), yerr=np.array(res_M)[:,1].std(axis=1), label='Moment est. simulation', c='orange')
ax[1].errorbar(Ms, np.array(res_M)[:,0].mean(axis=1), yerr=np.array(res_M)[:,0].std(axis=1), label='Plug-in est. simulation')
ax[1].plot(Ms, [true_shape_metric,] * len(Ms), label='True', c='k', ls='--')
ax[1].set_xlabel('# stimuli (M)')
ax[1].set_ylim(-0.1,1.1)
ax[1].set_title('Effect of sample size \n N=' + str(N_Msim))
ax[1].set_yticks(y_ticks)

ax[2].errorbar(Ns, np.array(res_N)[:,1].mean(axis=1), yerr=np.array(res_N)[:,1].std(axis=1), label='Moment est. simulation', c='orange')
ax[2].errorbar(Ns, np.array(res_N)[:,0].mean(axis=1), yerr=np.array(res_N)[:,0].std(axis=1), label='Plug-in est. simulation')
ax[2].plot(Ns, [true_shape_metric,] * len(Ns), label='True', c='k', ls='--')
ax[2].set_xlabel('# neurons (N)')
ax[2].set_ylim(-0.1,1.1)
ax[2].set_yticks(y_ticks)
ax[2].set_title('Effect of dimensionality \nM=' + str(M_Nsim))

ax[3].plot(np.arange(n_sims), [true_shape_metric,] * n_sims, label='True', c='k', ls='--')
ax[3].plot(np.arange(n_sims), res_ci[:,1], label='Moment est. ', c='orange')
ax[3].fill_between(np.arange(n_sims), res_ci[:,1] - ci_var_bias, res_ci[:,1] + ci_var_bias, alpha=0.5, color='orange', label='CI ' r'$(\alpha \leq 0.05)$')
ax[3].plot(np.arange(n_sims), res_ci[:,0], label='Plug-in est.')

ax[3].set_xlabel('Simulation')
ax[3].legend()
ax[3].set_title('Confidence intervals\n M=' + str(M_cisim) + ', N=' + str(N_cisim))
ax[3].set_yticks(y_ticks)
ax[3].set_ylim(-0.1,1.1)
for i in range(1,4):
    ax[i].set_yticklabels([])
    ax[i].set_ylabel('')
plt.tight_layout()
#save as a pdf in the figures folder
plt.savefig('./synth_sim_figs_temp.pdf', bbox_inches='tight')


#%% now make a simulation for calculating CI coverage
M_cisim = 200
N_cisim = 100
n_sims = 5000
true_shape_metric = 0.2
X, Y, singular_values, lam_x, lam_y, cov = src.sim_data(N_cisim, N_cisim, true_shape_metric, pl_slope, M_cisim, n_sims)
true_denom = (np.sum(lam_x)*np.sum(lam_y))**0.5
est_eigmoms = np.zeros((n_sims, num_moments+1))
for i in tqdm(range(n_sims)):
    est_eigmoms[i] = src.estimate_eigenmoments(X[i], Y[i], num_moments)
eig_mom_cov = np.cov(est_eigmoms.T)
sample_cov = np.matmul(np.transpose(X, (0, 2, 1)) , Y)
naive_est  = np.linalg.norm((sample_cov) / M_cisim, ord='nuc', axis=(1,2))/true_denom
gamma, bias, M_xt, xt = src.calc_gammas(
        X[0], Y[0], num_moments,        
        alpha=1.,
        eig_mom_cov=eig_mom_cov,
        max_eig = np.max(lam_x)**0.5, remove_constant=True,
        return_poly_basis=True,
        bias_frac=0.1)
est =  est_eigmoms @ gamma / true_denom
bias = bias/true_denom
var = gamma @ eig_mom_cov @ gamma / true_denom**2
ci_95 = 1.96*(var**0.5)
ci_var_bias = ci_95 + bias
res_ci = np.array([naive_est, est]).T

#calculate how often the ci contains the true value
ul, ll = res_ci[:,1] + ci_var_bias, res_ci[:,1] - ci_var_bias
ci_contains_true = np.logical_and(ll < true_shape_metric, ul > true_shape_metric)
print('CI contains true value ' + str(np.sum(ci_contains_true)/n_sims))