import jax
import jax.numpy as jnp
import numpy as np
from kernel_herding import kernel_herding_EIF, kernel_herding_plugin
from propensities import estimate_propensity
from kernels import gaussian_kernel
from utils import Array,Scalar
from KRR_utils import KRR,ard_median_heuristic,iso_median_heuristic,optimise,optimise_lambda, learn_hyperparams

def compute_EIF_a0(A_tr:Array,
                   M_tr:Array,
                   X_tr:Array,
                   Y_tr:Array,
                   A_te:Array,
                   M_te:Array,
                   X_te:Array,
                   Y_te:Array,
                   a0:int=1,
                   subsample:bool=True,
                   x_method:str='logistic',
                   mx_method:str='logistic',
                   params=None, 
                   explicit_m:bool=False, 
                   explicit_y:bool=False):
    """compute the one-step estimator for the counterfactual mean embedding under the front door criterion.
    We can't write the mean embedding in closed form as the feature map is infinite-dimensional, so instead we calculate
    two terms which, combined with the kernels on the training and test outcomes, can then be used to evalue MMDs, sample from the
    counterfactual, etc.
    
    setting explicit_m to True means we calculate the conditional mean embedding of m using 
        tabular methods instead of ridge regression. Only works if m and x are binary.
    setting explicit_y to True means calculating the conditional mean embeddings of m and y uing
        tabular methods instead of ridge regression. Integrals involving the mean embedding of y are
        then done using tabular methods as well. Only works if m and x are binary."""
    
    n_tr=Y_tr.shape[0]
    n_te=Y_te.shape[0]
    d_m=M_tr.shape[1]
    d_x=X_tr.shape[1]

    if params is None:
        print('learning hyperparameters')
        lambda_m,lambda_y,sigma_A,sigma_M,sigma_X=learn_hyperparams(A_tr,M_tr,X_tr,Y_tr,subsample=subsample)
    elif params is not None:
        lambda_m,lambda_y,sigma_A,sigma_M,sigma_X=params

    A_kernel = gaussian_kernel(sigma_A)
    M_kernel = gaussian_kernel(sigma_M)
    X_kernel = gaussian_kernel(sigma_X)

    #training-training kernel matrices
    K_AA=A_kernel.make_distance_matrix(A_tr,A_tr)
    K_MM=M_kernel.make_distance_matrix(M_tr,M_tr)
    K_XX=X_kernel.make_distance_matrix(X_tr,X_tr)

    #training-test kernel matrices
    K_Xx=X_kernel.make_distance_matrix(X_tr,X_te)
    K_Mm=M_kernel.make_distance_matrix(M_tr,M_te)
    K_Aa=A_kernel.make_distance_matrix(A_tr,A_te)

    #learn propensity and density ratio functions
    a_x =estimate_propensity(np.array(A_tr),np.array(X_tr),method=x_method)
    a_mx = estimate_propensity(np.array(A_tr),np.concatenate((np.array(X_tr),np.array(M_tr)),axis=1),method=mx_method)
    if explicit_m or explicit_y:
        m_ax = estimate_propensity(np.array(M_tr),np.concatenate((np.array(A_tr),np.array(X_tr)),axis=1),method=mx_method)
        a0_vec=a0*jnp.ones(A_te.shape)
        p_m_a0x_array=jnp.array(m_ax.predict_proba(np.concatenate((np.array(a0_vec),np.array(X_te)),axis=1)))
    
    if explicit_m and not explicit_y:
        K_M1=M_kernel.make_distance_matrix(M_tr,jnp.array([[1.]]))
        K_M0=M_kernel.make_distance_matrix(M_tr,jnp.array([[0.]]))

    
    
    K_A1=A_kernel.make_distance_matrix(A_tr,jnp.array([[1.]]))
    K_A0=A_kernel.make_distance_matrix(A_tr,jnp.array([[0.]]))
    K_Aa0=A_kernel.make_distance_matrix(A_tr,a0*jnp.ones(A_te.shape))

    #array of probabilities of each treatment a for each data point of shape (N,2)
    p_a_x_array=jnp.array(a_x.predict_proba(np.array(X_te)))
    p_a_mx_array=jnp.array(a_mx.predict_proba(np.concatenate((np.array(X_te),np.array(M_te)),axis=1)))
    
    #array of the probability of treatment a0|x for each data point of shape (N,)
    p_a0_x = a0*p_a_x_array[:,1] + (1-a0)*p_a_x_array[:,0]

    #array of the probability of observed treatment a|x for each data point of shape (N,)
    p_a_x = jnp.multiply(A_te.ravel(),p_a_x_array[:,1]) + jnp.multiply((jnp.ones(A_te.ravel().shape)-A_te.ravel()),p_a_x_array[:,0])
    
    #array of the probability of treatment a0|m,x for each data point of shape (N,)
    p_a0_mx = a0*p_a_mx_array[:,1] + (1-a0)*p_a_mx_array[:,0]

    #array of the probability of observed treatment a|m,x for each data point of shape (N,)
    p_a_mx = jnp.multiply(A_te.ravel(),p_a_mx_array[:,1]) + jnp.multiply((jnp.ones(A_te.ravel().shape)-A_te.ravel()),p_a_mx_array[:,0])
    
    #array of density ratios p(m|a0,x)/p(m|a,x) for each data point of shape (N,)
    dr_array = jnp.multiply(jnp.divide(p_a0_mx,p_a_mx),jnp.divide(p_a_x,p_a0_x))

    #tile this density ratio n times to form an NxN matrix where each row is the dr_array vector above
    dr_mat=jnp.tile(dr_array,(n_tr,1))

    #array for indicator term 1[a=a0]
    indicator_array = A_te.ravel() == a0*jnp.ones(A_te.ravel().shape)

    #matrix where each row is the array 1[a=a0]/p(a0|x)
    p_a0_x_reciprocal_mat=jnp.tile(jnp.divide(indicator_array,p_a0_x),(n_tr,1))

    #compute average K_Aa matrix given x
    K_Api=jnp.outer(K_A1,p_a_x_array[:,1])+jnp.outer(K_A0,p_a_x_array[:,0])
    #K_Api=K_AA@np.linalg.inv(K_XX+n_tr*lambda_a*np.eye(n_tr))@K_Xx

    K_AA_XX=jnp.multiply(K_AA,K_XX)
    K_AA_MM_XX=jnp.multiply(K_AA_XX,K_MM)
    R1=jnp.linalg.inv(K_AA_MM_XX+lambda_y*jnp.eye(n_tr))
    if explicit_m and not explicit_y:
        mu_m_a0x=jnp.outer(K_M1,p_m_a0x_array[:,1])+jnp.outer(K_M0,p_m_a0x_array[:,0])
    else:
        mu_m_a0x=K_MM@jnp.linalg.inv(K_AA_XX+lambda_m*jnp.eye(n_tr))@jnp.multiply(K_Aa0,K_Xx)
    K_Mm_Xx=jnp.multiply(K_Xx,K_Mm)
    
    
    if explicit_y:
        embedder=mu_y_amx()
        embedder.train(jnp.concatenate((A_tr,M_tr,X_tr),axis=1))
        first_term_mat=jnp.multiply(dr_mat,embedder.predict(jnp.concatenate((A_te,M_te,X_te),axis=1)))
    else:
        first_term_mat=R1@jnp.multiply(dr_mat,jnp.multiply(K_Aa,K_Mm_Xx))
        
    first_term_v2=1/n_te * first_term_mat.sum(axis=1).reshape(-1,1)

    if explicit_y:
        mu_y_1mx=embedder.predict(jnp.concatenate((jnp.ones(A_te.shape),M_te,X_te),axis=1))
        mu_y_0mx=embedder.predict(jnp.concatenate((jnp.zeros(A_te.shape),M_te,X_te),axis=1))

        mu_y_a1x=embedder.predict(jnp.concatenate((A_te,jnp.ones(M_te.shape),X_te),axis=1))
        mu_y_a0x=embedder.predict(jnp.concatenate((A_te,jnp.zeros(M_te.shape),X_te),axis=1))

        mu_y_11x=embedder.predict(jnp.concatenate((jnp.ones(A_te.shape),jnp.ones(M_te.shape),X_te),axis=1))
        mu_y_01x=embedder.predict(jnp.concatenate((jnp.zeros(A_te.shape),jnp.ones(M_te.shape),X_te),axis=1))
        mu_y_10x=embedder.predict(jnp.concatenate((jnp.ones(A_te.shape),jnp.zeros(M_te.shape),X_te),axis=1))
        mu_y_00x=embedder.predict(jnp.concatenate((jnp.zeros(A_te.shape),jnp.zeros(M_te.shape),X_te),axis=1))

        p_1_x_mat = jnp.tile(p_a_x_array[:,1],(n_tr,1))
        p_0_x_mat = jnp.tile(p_a_x_array[:,0],(n_tr,1))
        
        p_1_a0x_mat = jnp.tile(p_m_a0x_array[:,1],(n_tr,1))
        p_0_a0x_mat = jnp.tile(p_m_a0x_array[:,0],(n_tr,1))

        mu_y_mu_a = jnp.multiply(mu_y_1mx,p_1_x_mat) + jnp.multiply(mu_y_0mx, p_0_x_mat)
        mu_y_mu_m = jnp.multiply(mu_y_a1x,p_1_a0x_mat) + jnp.multiply(mu_y_a0x,p_0_a0x_mat)

        mu_y_mu_a_mu_m = (jnp.multiply(mu_y_11x,jnp.multiply(p_1_a0x_mat,p_1_x_mat)) + 
                          jnp.multiply(mu_y_10x,jnp.multiply(p_0_a0x_mat,p_1_x_mat)) + 
                          jnp.multiply(mu_y_01x,jnp.multiply(p_1_a0x_mat,p_0_x_mat)) + 
                          jnp.multiply(mu_y_00x,jnp.multiply(p_0_a0x_mat,p_0_x_mat)))
        
        second_term_mat = jnp.multiply(p_a0_x_reciprocal_mat,(mu_y_mu_a-mu_y_mu_a_mu_m))

    else:
        second_term_mat = R1@(jnp.multiply(p_a0_x_reciprocal_mat,(jnp.multiply(K_Api,K_Mm_Xx)-jnp.multiply(jnp.multiply(K_Api,mu_m_a0x),K_Xx))))
    second_term_v2=1/n_te * second_term_mat.sum(axis=1).reshape(-1,1)

    if explicit_y:
        third_term_mat=mu_y_mu_m
        
    else:
        third_term_mat=R1@jnp.multiply(jnp.multiply(K_Aa,mu_m_a0x),K_Xx)
    third_term_v2=1/n_te * third_term_mat.sum(axis=1).reshape(-1,1)
    
    EIF_estimate=second_term_v2+third_term_v2-first_term_v2

    return dr_array/n_te,EIF_estimate

def compute_plugin_a0(A_tr:Array,
                      M_tr:Array,
                      X_tr:Array,
                      Y_tr:Array,
                      a0:int=1,
                      subsample:bool=False,
                      x_method:str='logistic',
                      params=None,
                      explicit_m:bool=False,
                      explicit_y:bool=False):
    """compute the plug-in estimator for the counterfactual mean embedding under the front door criterion.
    We can't write the mean embedding in closed form as the feature map is infinite-dimensional, so instead we calculate
    a term which, combined with the kernel on the training outcomes, can then be used to evalue MMDs, sample from the
    counterfactual, etc.
    
    setting explicit_m to True means we calculate the conditional mean embedding of m using 
        tabular methods instead of ridge regression. Only works if m and x are binary.
    setting explicit_y to True means calculating the conditional mean embeddings of m and y uing
        tabular methods instead of ridge regression. Integrals involving the mean embedding of y are
        then done using tabular methods as well. Only works if m and x are binary."""
        
    n_tr=Y_tr.shape[0]
    d_m=M_tr.shape[1]
    d_x=X_tr.shape[1]
    #compute kernel bandwidths (median heuristic) and resulting kernels
    
    if params is None:
        print('learning hyperparameters')
        lambda_m,lambda_y,sigma_A,sigma_M,sigma_X=learn_hyperparams(A_tr,M_tr,X_tr,Y_tr,subsample=subsample)
    elif params is not None:
        lambda_m,lambda_y,sigma_A,sigma_M,sigma_X=params

    A_kernel = gaussian_kernel(sigma_A)
    K_AA=A_kernel.make_distance_matrix(A_tr,A_tr)
    M_kernel = gaussian_kernel(sigma_M)
    K_MM=M_kernel.make_distance_matrix(M_tr,M_tr)
    X_kernel = gaussian_kernel(sigma_X)
    K_XX=X_kernel.make_distance_matrix(X_tr,X_tr)

    K_AA_XX=jnp.multiply(K_AA,K_XX)
    K_AA_MM_XX=jnp.multiply(K_AA_XX,K_MM)

    #learn propensity and density ratio functions
    a_x =estimate_propensity(np.array(A_tr),np.array(X_tr),method=x_method)
    if explicit_m or explicit_y:
        m_ax = estimate_propensity(np.array(M_tr),np.concatenate((np.array(A_tr),np.array(X_tr)),axis=1),method=x_method)
        a0_vec=a0*jnp.ones(A_tr.shape)
        p_m_a0x_array=jnp.array(m_ax.predict_proba(np.concatenate((np.array(a0_vec),np.array(X_tr)),axis=1)))
    
    if explicit_m:
        K_M1=M_kernel.make_distance_matrix(M_tr,jnp.array([[1.]]))
        K_M0=M_kernel.make_distance_matrix(M_tr,jnp.array([[0.]]))
        
    #a_mx = estimate_propensity(np.array(A_tr),np.concatenate((np.array(X_tr),np.array(M_tr)),axis=1))
    
    K_A1=A_kernel.make_distance_matrix(A_tr,jnp.array([[1.]]))
    K_A0=A_kernel.make_distance_matrix(A_tr,jnp.array([[0.]]))
    K_Aa0=A_kernel.make_distance_matrix(A_tr,a0*jnp.ones(A_tr.shape))


    p_a_x_array=jnp.array(a_x.predict_proba(np.array(X_tr)))
    
    

    #compute average K_Aa matrix given x
    K_Api=jnp.outer(K_A1,p_a_x_array[:,1])+jnp.outer(K_A0,p_a_x_array[:,0])
    #K_Api=K_AA@np.linalg.inv(K_XX+n_tr*lambda_a*np.eye(n_tr))@K_XX

    R1=jnp.linalg.inv(K_AA_MM_XX+lambda_y*jnp.eye(n_tr))
    
    if explicit_m:
        mu_m_a0x=jnp.outer(K_M1,p_m_a0x_array[:,1])+jnp.outer(K_M0,p_m_a0x_array[:,0])
    else:
        mu_m_a0x=K_MM@jnp.linalg.inv(K_AA_XX+lambda_m*jnp.eye(n_tr))@jnp.multiply(K_Aa0,K_XX)
    
    if explicit_y:
        embedder=mu_y_amx()
        embedder.train(jnp.concatenate((A_tr,M_tr,X_tr),axis=1))
        mu_y_11x=embedder.predict(jnp.concatenate((jnp.ones(A_tr.shape),jnp.ones(M_tr.shape),X_tr),axis=1))
        mu_y_01x=embedder.predict(jnp.concatenate((jnp.zeros(A_tr.shape),jnp.ones(M_tr.shape),X_tr),axis=1))
        mu_y_10x=embedder.predict(jnp.concatenate((jnp.ones(A_tr.shape),jnp.zeros(M_tr.shape),X_tr),axis=1))
        mu_y_00x=embedder.predict(jnp.concatenate((jnp.zeros(A_tr.shape),jnp.zeros(M_tr.shape),X_tr),axis=1))
        
        p_1_x_mat = jnp.tile(p_a_x_array[:,1],(n_tr,1))
        p_0_x_mat = jnp.tile(p_a_x_array[:,0],(n_tr,1))
        
        p_1_a0x_mat = jnp.tile(p_m_a0x_array[:,1],(n_tr,1))
        p_0_a0x_mat = jnp.tile(p_m_a0x_array[:,0],(n_tr,1))

        second_term_mat = (jnp.multiply(mu_y_11x,jnp.multiply(p_1_a0x_mat,p_1_x_mat)) + 
                          jnp.multiply(mu_y_10x,jnp.multiply(p_0_a0x_mat,p_1_x_mat)) + 
                          jnp.multiply(mu_y_01x,jnp.multiply(p_1_a0x_mat,p_0_x_mat)) + 
                          jnp.multiply(mu_y_00x,jnp.multiply(p_0_a0x_mat,p_0_x_mat)))
        
    else:
        second_term_mat= R1@jnp.multiply(jnp.multiply(K_Api,mu_m_a0x),K_XX)
    second_term_v2=1/n_tr * second_term_mat.sum(axis=1).reshape(-1,1)

    return second_term_v2

def MMD(X:Array,
        Y:Array):
    
    #MMD based on samples X and Y from distributions P and Q
    sigma=ard_median_heuristic(Y)
    kernel=gaussian_kernel(sigma)
    K_tr_tr = kernel.make_distance_matrix(X,X)
    K_tr_te = kernel.make_distance_matrix(X,Y)
    K_te_te = kernel.make_distance_matrix(Y,Y)
    first_term = K_tr_tr.mean()
    second_term = K_tr_te.mean() + K_tr_te.mean()
    third_term = K_te_te.mean()
    return first_term - second_term + third_term
    
def MMD_plugin(Y_tr:Array,
               Y_truth:Array,
               plugin_estimate:Array):
    
    #MMD based on plugin mean embedding for distribution P and sample from distribution Q
    sigma=ard_median_heuristic(Y_tr)
    kernel=gaussian_kernel(sigma)
    K_tr_tr = kernel.make_distance_matrix(Y_tr, Y_tr)
    K_tr_te = kernel.make_distance_matrix(Y_tr,Y_truth)
    K_te_te = kernel.make_distance_matrix(Y_truth, Y_truth)
    first_term = (plugin_estimate.T @ K_tr_tr @ plugin_estimate).mean()
    second_term = (plugin_estimate.T @ K_tr_te).mean() + (K_tr_te.T @ plugin_estimate).mean()
    third_term = (K_te_te).mean()
    return first_term - second_term + third_term
    
def MMD_dr(Y_te:Array,
           Y_tr:Array,
           Y_truth:Array,
           EIF_estimate:Array,
           dr:Array):
    
    #MMD based on doubly robust mean embedding for distribution P and sample from distribution Q
    sigma=ard_median_heuristic(Y_tr)
    kernel=gaussian_kernel(sigma)
    
    K_tr_tr = kernel.make_distance_matrix(Y_tr, Y_tr)
    K_tr_te = kernel.make_distance_matrix(Y_tr,Y_te)
    K_te_te = kernel.make_distance_matrix(Y_te,Y_te)
    K_truth_truth = kernel.make_distance_matrix(Y_truth,Y_truth)
    K_tr_truth=kernel.make_distance_matrix(Y_tr,Y_truth)
    K_te_truth=kernel.make_distance_matrix(Y_te,Y_truth)
    first_term = (EIF_estimate.T @ K_tr_tr @ EIF_estimate).mean() + (dr.T @ K_te_te @ dr).mean() + (EIF_estimate.T @ K_tr_te @ dr).mean() + (dr.T @ K_tr_te.T @ EIF_estimate).mean()
    second_term = (dr.T @ K_te_truth).mean() + (K_te_truth.T @ dr).mean() + (EIF_estimate.T @ K_tr_truth).mean() + (K_tr_truth.T @ EIF_estimate).mean()
    third_term = K_truth_truth.mean()
    return first_term - second_term + third_term

def KTE(A_1,M_1,X_1,Y_1,A_2,M_2,X_2,Y_2,x_method='logistic',mx_method='logistic',subsample=True,params=None):
    """computes the test statistic from the doubly robust kernel treatment effects test"""
    
    n_1=Y_1.shape[0]
    n_2=Y_2.shape[0]
    assert n_1==n_2
    
    #compute inner product
    kernel=gaussian_kernel(ard_median_heuristic(Y_1))
    K_11=kernel.make_distance_matrix(Y_1,Y_1)
    K_22=kernel.make_distance_matrix(Y_2,Y_2)
    K_12=kernel.make_distance_matrix(Y_1,Y_2)
    K_21=K_12.T

    def compute_EIF(A_tr,M_tr,X_tr,Y_tr,A_te,M_te,X_te,Y_te,subsample=True,x_method='logistic',mx_method='logistic',params=None):
        n_tr=Y_tr.shape[0]
        n_te=Y_te.shape[0]
        d_m=M_tr.shape[1]
        d_x=X_tr.shape[1]
    
        #compute kernel bandwidths (median heuristic) and resulting kernels
        AX=jnp.concatenate((A_tr,X_tr),axis=1)
        AMX=jnp.concatenate((A_tr,M_tr,X_tr),axis=1)
        
        if params is None:
            lambda_m,lambda_y,sigma_A,sigma_M,sigma_X=learn_hyperparams(A_tr,M_tr,X_tr,Y_tr,subsample=subsample)
        elif params is not None:
            lambda_m,lambda_y,sigma_A,sigma_M,sigma_X=params

        A_kernel = gaussian_kernel(sigma_A)
        K_AA=A_kernel.make_distance_matrix(A_tr,A_tr)
        M_kernel = gaussian_kernel(sigma_M)
        K_MM=M_kernel.make_distance_matrix(M_tr,M_tr)
        X_kernel = gaussian_kernel(sigma_X)
        K_XX=X_kernel.make_distance_matrix(X_tr,X_tr)

        #learn propensity and density ratio functions
        a_x =estimate_propensity(np.array(A_tr),np.array(X_tr),method=x_method)
        a_mx = estimate_propensity(np.array(A_tr),np.concatenate((np.array(X_tr),np.array(M_tr)),axis=1),method=mx_method)
    
        K_A1=A_kernel.make_distance_matrix(A_tr,jnp.array([[1.]]))
        K_A0=A_kernel.make_distance_matrix(A_tr,jnp.array([[0.]]))

        #array of probabilities of each treatment a for each data point of shape (N,2)
        p_a_x_array=jnp.array(a_x.predict_proba(np.array(X_te)))
        p_a_mx_array=jnp.array(a_mx.predict_proba(np.concatenate((np.array(X_te),np.array(M_te)),axis=1)))

        #array of the probability of treatment a0|x for each data point of shape (N,1)
        p_0_x = p_a_x_array[:,0]
        p_1_x = p_a_x_array[:,1]

        #array of the probability of observed treatment a|x for each data point of shape (N,1)
        p_a_x = jnp.multiply(A_te.ravel(),p_a_x_array[:,1]) + jnp.multiply((jnp.ones(A_te.ravel().shape)-A_te.ravel()),p_a_x_array[:,0])

        #array of the probability of treatment a0|m,x for each data point of shape (N,1)
        p_0_mx = p_a_mx_array[:,0]
        p_1_mx = p_a_mx_array[:,1]

        #array of the probability of observed treatment a|m,x for each data point of shape (N,1)
        p_a_mx = jnp.multiply(A_te.ravel(),p_a_mx_array[:,1]) + jnp.multiply((jnp.ones(A_te.ravel().shape)-A_te.ravel()),p_a_mx_array[:,0])

        #array of density ratios p(m|a0,x)/p(m|a,x) for each data point of shape (N,1)
        dr_0_array = jnp.multiply(jnp.divide(p_0_mx,p_a_mx),jnp.divide(p_a_x,p_0_x))
        dr_1_array = jnp.multiply(jnp.divide(p_1_mx,p_a_mx),jnp.divide(p_a_x,p_1_x))

        #tile this density ratio n times to form an NxN matrix where each row is the dr_array vector above
        dr_0_mat=jnp.tile(dr_0_array,(n_tr,1))
        dr_1_mat=jnp.tile(dr_1_array,(n_tr,1))

        #array for indicator term 1[a=a0]
        indicator_0_array = A_te.ravel() == jnp.zeros(A_te.ravel().shape)
        indicator_1_array = A_te.ravel() == jnp.ones(A_te.ravel().shape)

        #matrix where each row is the array 1[a=a0]/p(a0|x)
        p_0_x_reciprocal_mat=jnp.tile(jnp.divide(indicator_0_array,p_0_x),(n_tr,1))
        p_1_x_reciprocal_mat=jnp.tile(jnp.divide(indicator_1_array,p_1_x),(n_tr,1))

        #compute K_Xx matrix
        K_Xx=X_kernel.make_distance_matrix(X_tr,X_te)
        #compute K_Mm matrix
        K_Mm=M_kernel.make_distance_matrix(M_tr,M_te)
        #compute K_Aa matrix
        K_Aa=A_kernel.make_distance_matrix(A_tr,A_te)
    

        #compute average K_Aa matrix given x
        K_Api=jnp.outer(K_A1,p_a_x_array[:,1])+jnp.outer(K_A0,p_a_x_array[:,0])

        K_AA_XX=jnp.multiply(K_AA,K_XX)
        K_AA_MM_XX=jnp.multiply(K_AA_XX,K_MM)
        R1=jnp.linalg.inv(K_AA_MM_XX+lambda_y*jnp.eye(n_tr))
        mu_m_0x=K_MM@jnp.linalg.inv(K_AA_XX+lambda_m*jnp.eye(n_tr))@jnp.multiply(K_A0,K_Xx)
        mu_m_1x=K_MM@jnp.linalg.inv(K_AA_XX+lambda_m*jnp.eye(n_tr))@jnp.multiply(K_A1,K_Xx)
        
        K_Mm_Xx=jnp.multiply(K_Xx,K_Mm)
    
        first_term_0_mat=R1@jnp.multiply(dr_0_mat,jnp.multiply(K_Aa,K_Mm_Xx))
        first_term_1_mat=R1@jnp.multiply(dr_1_mat,jnp.multiply(K_Aa,K_Mm_Xx))

        second_term_0_mat = R1@(jnp.multiply(p_0_x_reciprocal_mat,(jnp.multiply(K_Api,K_Mm_Xx)-jnp.multiply(jnp.multiply(K_Api,mu_m_0x),K_Xx))))
        second_term_1_mat = R1@(jnp.multiply(p_1_x_reciprocal_mat,(jnp.multiply(K_Api,K_Mm_Xx)-jnp.multiply(jnp.multiply(K_Api,mu_m_1x),K_Xx))))

        third_term_0_mat=R1@jnp.multiply(jnp.multiply(K_Aa,mu_m_0x),K_Xx)
        third_term_1_mat=R1@jnp.multiply(jnp.multiply(K_Aa,mu_m_1x),K_Xx)

        EIF_0_mat=second_term_0_mat+third_term_0_mat-first_term_0_mat
        EIF_1_mat=second_term_1_mat+third_term_1_mat-first_term_1_mat

        EIF_diff_mat=EIF_1_mat-EIF_0_mat
        dr_diff_array=dr_1_array-dr_0_array
        
        return dr_diff_array,EIF_diff_mat

    #train a doubly robust estimator using 1 as 'tr' and 2 as 'te' for both a0=1 and a0=0
    dr_1_diff,eif_1_diff=compute_EIF(A_1,M_1,X_1,Y_1,A_2,M_2,X_2,Y_2,x_method=x_method,mx_method=mx_method,params=params)
    dr_1_mat=jnp.tile(dr_1_diff,(n_2,1)).T

    #train a doubly robust estimator using 2 as 'tr' and 1 as 'te' for both a0=1 and a0=0
    dr_2_diff,eif_2_diff=compute_EIF(A_2,M_2,X_2,Y_2,A_1,M_1,X_1,Y_1,x_method=x_method,mx_method=mx_method,params=params)
    dr_2_mat=jnp.tile(dr_2_diff,(n_1,1)).T

    term_1=jnp.multiply(dr_1_mat,jnp.multiply(K_21,dr_2_mat.T)) 
    term_2=eif_1_diff.T@K_12@eif_2_diff
    term_3=jnp.multiply(eif_1_diff.T@K_11,dr_2_mat.T)
    term_4=jnp.multiply(dr_1_mat,K_22@eif_2_diff)

    f_mat=term_1+term_2+term_3+term_4
    f_vec=f_mat.mean(axis=1)
    s=f_vec.std()
    f=f_vec.mean()
    T=jnp.sqrt(n_1)*f/s
    
    return T

class mu_y_amx:

    def __init__(self):
        self.opts = jnp.array([[0,0,0],
                          [0,1,0],
                          [1,0,0],
                          [1,1,0],
                          [0,0,1],
                          [0,1,1],
                          [1,0,1],
                          [1,1,1]])
        

    def train(self,X:Array):
        #learn the alpha vectors from data
        n=X.shape[0]
        self.alphas=jnp.zeros((n,8))
        for i in range(8):
            x_val=self.opts[i,:]
            indices=jnp.where(jnp.all(X==x_val,axis=1))[0]
            num=indices.size
            if num != 0:
                self.alphas=self.alphas.at[indices,i].set(1./num)
        return self.alphas
    
    def predict(self,X:Array):
        @jax.jit
        def one_step(X:Array,i:Scalar):
            alph = inner(self.opts,self.alphas,X[i,:]).ravel()
            return X, alph

        _,res=jax.lax.scan(one_step,X,jnp.arange(X.shape[0]))
        return res.T

@jax.jit
def inner(opts,alphas:Array,X:Array):
    #output the correct alpha vector given input X
    index = jnp.where(jnp.all(opts==X,axis=1),size=1)
    alpha = alphas[:,index]
    return alpha
