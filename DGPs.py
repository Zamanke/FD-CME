import jax
import jax.numpy as jnp
from utils import Array, Scalar

def DGP_binary(n: int,
               seed: int =0,
               dim:int =1,
               gt:bool =False,
               a0:int=1,
               u_scale:float=0, 
               null:bool=False) -> Array:
    """univariate distribution with a mean shift;
    A, M and X are binary; Y is continuous;
    dim is a dummy variable and has no effect"""
    # Give each structural noise its own key, including under intervention.
    key_x,key_u,key_a,key_m,key_y=jax.random.split(jax.random.key(seed),5)

    if null:
        k=0
    else:
        k=2
    X=jax.random.binomial(key_x,1,0.5,shape=(n,1))

    U=u_scale*jax.random.normal(key_u,shape=(n,1))
    
    if gt:
        A=a0*jnp.ones((n,1))
    else:
        logit_matrix=2*X-1+U
        prob_matrix=jnp.pow(jnp.ones((n,1))+jnp.exp(logit_matrix),-1)
        A=jax.random.binomial(key_a,1,prob_matrix)

    logit_matrix=k*A+2*X-2
    prob_matrix=jnp.pow(jnp.ones((n,1))+jnp.exp(logit_matrix),-1)
    M=jax.random.binomial(key_m,1,prob_matrix)

    Y=M+X+U+jax.random.normal(key_y,shape=(n,1))
    return A,M,X,Y

def DGP_univariate(n: int,
               seed: int =0,
               dim:int =1,
               gt:bool =False,
               a0:int=1,
               u_scale:float=1, 
               null:bool=False) -> Array:
    """univariate distribution with a mean shift; dim is a dummy variable and has no effect"""
    if null:
        k=0
    else:
        k=1
        
    key_x,key_u,key_a,key_m,key_y=jax.random.split(jax.random.key(seed),5)

    X=jax.random.normal(key_x,shape=(n,1))

    U=u_scale*jax.random.normal(key_u,shape=(n,1))
    
    if gt:
        A=a0*jnp.ones((n,1))
    else:
        logit_matrix=3*X+U
        prob_matrix=jnp.pow(jnp.ones((n,1))+jnp.exp(logit_matrix),-1)
        A=jax.random.binomial(key_a,1,prob_matrix)

    M=(3*jnp.ones((n,1))-2.5*k*A)*jax.random.normal(key_m,shape=(n,1))+X+3*k*A

    Y=M+X+U+jax.random.normal(key_y,shape=(n,1))
    return A,M,X,Y


def DGP_multimodal(n: int,
               seed: int =0,
               dim:int =1,
               gt:bool =False,
               a0:int=1,
               u_scale:float=1, 
               null:bool=False) -> Array:
    """unimodal distribution for a0=0 and bimodal distribution for a0=1, with approximately the same mean. 
    Always univariate; dim is a dummy variable"""
    key_x,key_u,key_a,key_m,key_y=jax.random.split(jax.random.key(seed),5)
    
    if null:
        k=0
    else:
        k=1

    X=jax.random.normal(key_x,shape=(n,1))

    U=u_scale*jax.random.normal(key_u,shape=(n,1))

    
    if gt:
        A=a0*jnp.ones((n,1))
    else:
        logit_matrix=3*X+U
        prob_matrix=jnp.pow(jnp.ones((n,1))+jnp.exp(logit_matrix),-1)
        A=jax.random.binomial(key_a,1,prob_matrix)

    M=jnp.sign(X[:,0]).reshape(-1,1)*((3*jnp.ones((n,1))-2.5*k*A)*jax.random.normal(key_m,shape=(n,1))+X+3*k*A)

    Y=M+X+U+jax.random.normal(key_y,shape=(n,1))
    return A,M,X,Y

def DGP_high_dim_X(n: int,
               seed: int =0,
               dim:int =1,
               gt:bool =False,
               a0:int=1,
               u_scale:float=1, 
               null:bool=False) -> Array:
    """arbitrarily high-dimensional covariates X; all other variables are 1D.
    Treatment causes changes in mean, variance, skewness, etc"""
    D=jnp.hstack((jnp.zeros((dim,1)),jnp.vstack((jnp.eye(dim-1),jnp.zeros((dim-1,))))))
    cov = jnp.eye(dim)+0.5*D+0.5*D.T
    beta = jnp.pow(jnp.arange(1.,dim+1),-2).reshape(-1,1)

    if null:
        k=0
    else:
        k=1
        
    key_x,key_u,key_a,key_m,key_y=jax.random.split(jax.random.key(seed),5)

    X=jax.random.multivariate_normal(key_x,jnp.zeros((dim,)),cov,shape=(n,))

    U=u_scale*jax.random.normal(key_u,shape=(n,1))
    
    if gt:
        A=a0*jnp.ones((n,1))
    else:
        logit_matrix=3*X@beta+U
        prob_matrix=jnp.pow(jnp.ones((n,1))+jnp.exp(logit_matrix),-1)
        A=jax.random.binomial(key_a,1,prob_matrix)

    
    M=jnp.sign(X[:,0]).reshape(-1,1)*((3*jnp.ones((n,1))-2.5*k*A)*jax.random.normal(key_m,shape=(n,1))+X@beta+3*k*A)

    Y=M+X@beta+U+jax.random.normal(key_y,shape=(n,1))
    return A,M,X,Y

def DGP_high_dim_M(n: int,
               seed: int =0,
               dim:int =1,
               gt:bool =False,
               a0:int=1,
               u_scale:float=1, 
               null:bool=False) -> Array:
    """3 dimensional mediator M and all other variables are univariate. dim is a dummy variable"""
    dim=3
    key_x,key_u,key_a,key_m,key_y=jax.random.split(jax.random.key(seed),5)
    D=jnp.hstack((jnp.zeros((dim,1)),jnp.vstack((jnp.eye(dim-1),jnp.zeros((dim-1,))))))
    cov = jnp.eye(dim)+0.5*D+0.5*D.T
    beta = jnp.pow(jnp.arange(1.,dim+1),-2).reshape(-1,1)
    if null:
        k=0
    else:
        k=1
    
    X=jax.random.normal(key_x,shape=(n,1))

    U=u_scale*jax.random.normal(key_u,shape=(n,1))

    
    if gt:
        A=a0*jnp.ones((n,1))
    else:
        logit_matrix=3*X+U
        prob_matrix=jnp.pow(jnp.ones((n,1))+jnp.exp(logit_matrix),-1)
        A=jax.random.binomial(key_a,1,prob_matrix)


    mean=jnp.concatenate((5*A*jnp.pow(X,2)+3*k*A,3*X,2*k*A*X),axis=1)

    M=jax.random.multivariate_normal(key_m,mean,cov,shape=n)

        
    Y=3*X+5*M@beta+2*U+jax.random.normal(key_y,shape=(n,1))
    return A,M,X,Y


def DGP_high_dim_MX(n: int,
               seed: int =0,
               dim:int =1,
               gt:bool =False,
               a0:int=1,
               u_scale:float=1, 
               null:bool=False) -> Array:
    if null:
        k=0
    else:
        k=1
    """can set the dimension of M and X (they will always have the same dimension)"""
    D=jnp.hstack((jnp.zeros((dim,1)),jnp.vstack((jnp.eye(dim-1),jnp.zeros((dim-1,))))))
    cov = jnp.eye(dim)+0.5*D+0.5*D.T
    beta = jnp.pow(jnp.arange(1.,dim+1),-2).reshape(-1,1)
    key_x,key_u,key_a,key_m,key_y=jax.random.split(jax.random.key(seed),5)
    
    X=jax.random.multivariate_normal(key_x,jnp.zeros((dim,)),cov,shape=(n,))

    U=u_scale*jax.random.normal(key_u,shape=(n,1))
    
    if gt:
        A=a0*jnp.ones((n,1))
    else:
        logit_matrix=3*X@beta+U
        prob_matrix=jnp.pow(jnp.ones((n,1))+jnp.exp(logit_matrix),-1)
        A=jax.random.binomial(key_a,1,prob_matrix)

    mean=5*A*jnp.pow(X,2)+3*k*A+3*X+2*k*A*X

    M=jax.random.multivariate_normal(key_m,mean,cov,shape=n)

        
    Y=3*X@beta+5*M@beta+2*U+jax.random.normal(key_y,shape=(n,1))
    return A,M,X,Y

def DGP_high_dim_Y(n: int,
               seed: int =0,
               dim:int =1,
               gt:bool =False,
               a0:int=1,
               u_scale:float=1, 
               null:bool=False) -> Array:
    """can set the dimension of M and X; Y will be 2-dimensional"""

    if null:
        k=0
    else:
        k=1
    D=jnp.hstack((jnp.zeros((dim,1)),jnp.vstack((jnp.eye(dim-1),jnp.zeros((dim-1,))))))
    cov = jnp.eye(dim)+0.5*D+0.5*D.T
    beta = jnp.pow(jnp.arange(1.,dim+1),-2).reshape(-1,1)

    D_2=jnp.hstack((jnp.zeros((2,1)),jnp.vstack((jnp.eye(2-1),jnp.zeros((2-1,))))))
    cov_2 = jnp.eye(2)+0.5*D_2+0.5*D_2.T
    
    key_x,key_u,key_a,key_m,key_y=jax.random.split(jax.random.key(seed),5)

    
    X=jax.random.multivariate_normal(key_x,jnp.zeros((dim,)),cov,shape=(n,))

    U=u_scale*jax.random.normal(key_u,shape=(n,1))
    
    if gt:
        A=a0*jnp.ones((n,1))
    else:
        logit_matrix=3*X@beta+U
        prob_matrix=jnp.pow(jnp.ones((n,1))+jnp.exp(logit_matrix),-1)
        A=jax.random.binomial(key_a,1,prob_matrix)

    mean=5*A*jnp.pow(X,2)+3*k*A+3*X+2*k*A*X

    M=jnp.sign(X[:,0]).reshape(-1,1)*jax.random.multivariate_normal(key_m,mean,cov,shape=n)

        
    m_2=jnp.concatenate((3*X@beta,5*M@beta+2*U),axis=1)+10*jnp.ones((n,2))
    
    Y=jnp.sign(M[:,0]).reshape(-1,1)*jax.random.multivariate_normal(key_y,m_2,cov_2,shape=n)
    return A,M,X,Y
