import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from DGPs import DGP_univariate, DGP_high_dim_X, DGP_high_dim_M, DGP_high_dim_MX, DGP_high_dim_Y, DGP_binary

from utils import Array,Scalar
from KRR_utils import learn_hyperparams
from embeddings import compute_EIF_a0, compute_plugin_a0, MMD, MMD_plugin, MMD_dr, KTE

def run_convergence_experiment(DGP,
                               n:int,
                               seed:int=0,
                               dim:int=1,
                               true_data:Array=None,
                               params=None,
                               subsample:bool=True,
                               treatment:int=1,
                               x_method:str='logistic',
                               mx_method:str='logistic',
                               m:bool=False,
                               y:bool=False):
    """For a given data generating process, run the plugin and doubly-robust estimators and 
    compute the MMD between them and the ground truth distribution. If no ground truth is given,
    it will be generated. Variable 'm' is passed to 'explicit_m' in the estimating functions, and 
    'y' is passed to 'explicit_y'.
    Available propensity score estimation techniques are tabular ('table'), logistic regresion
    ('logistic'), gradient-boosted machine ('reggbm'), histogram gradient-boosted machine 
    ('histgbm'), or random forest ('forest')."""
    
    A,M,X,Y = DGP(2*n,seed=seed,dim=dim)
    A_tr=A[:n,:]
    A_te=A[n:,:]
    M_tr=M[:n,:]
    M_te=M[n:,:]
    X_tr=X[:n,:]
    X_te=X[n:,:]
    Y_tr=Y[:n,:]
    Y_te=Y[n:,:]

    if true_data is None:
        print('Generating ground truth data')
        A_truth,M_truth,X_truth,Y_truth = DGP(10000,seed=101,dim=dim,gt=True,a0=treatment)
    else:
        A_truth,M_truth,X_truth,Y_truth=true_data

    if params is None:
        print('Learning hyperparameters')
        params=learn_hyperparams(A_tr,M_tr,X_tr,Y_tr,subsample=subsample)

    dr,EIF = compute_EIF_a0(A_tr,M_tr,X_tr,Y_tr,A_te,M_te,X_te,Y_te,a0=treatment,x_method=x_method,mx_method=mx_method,params=params,explicit_m=m, explicit_y=y)
    div_dr = jnp.sqrt(MMD_dr(Y_te,Y_tr,Y_truth,EIF,dr))

    plugin = compute_plugin_a0(A_tr,M_tr,X_tr,Y_tr,a0=treatment,x_method=x_method,params=params,explicit_m=m, explicit_y=y)
    div_pi = jnp.sqrt(MMD_plugin(Y_tr,Y_truth,plugin))
    return div_dr, div_pi