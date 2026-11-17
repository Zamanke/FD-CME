from typing import Any, Callable, Type, Optional
from typing_extensions import TypeAlias, Self
import jax
import optax
import jax.numpy as jnp
from jax import vmap,grad,random
from flax import struct
import numpy as np
import abc
from functools import partial
from jax.tree_util import tree_map

Array: TypeAlias = Any
Scalar: TypeAlias = Any
Distribution: TypeAlias = Any

class base_kernel(struct.PyTreeNode, metaclass=abc.ABCMeta):
    #from https://github.com/hudsonchen/DrMMD/blob/main/kwgflows/divergences/mmd.py
    @abc.abstractmethod
    def __call__(self, x: Array, y: Array) -> Array:
        #compute kernel between data points x and y
        raise NotImplementedError

    def make_distance_matrix(self, X: Array, Y: Array) -> Array:
        #makes kernel matrix
        return vmap(vmap(type(self).__call__, (None, None, 0)), (None, 0, None))(
            self, X, Y)

class rkhs_element(struct.PyTreeNode):
    # from https://github.com/hudsonchen/DrMMD/blob/main/kwgflows/divergences/mmd.py
    X: Array
    w: Array
    kernel: base_kernel

    def __add__(self, other: Self):
        assert self.kernel is other.kernel
        return type(self)(
            X=jnp.concatenate([self.X, other.X], axis=0),
            w=jnp.concatenate([self.w, other.w], axis=0),
            kernel=self.kernel,
        )

    def __sub__(self, other: Self):
        assert self.kernel is other.kernel
        return type(self)(
            X=jnp.concatenate([self.X, other.X], axis=0),
            w=jnp.concatenate([self.w, -other.w], axis=0),
            kernel=self.kernel,
        )

    def __mul__(self, other: Array):
        # other should be scalar
        assert other.ndim == 0
        return self.replace(w=self.w * other)

    def __neg__(self):
        return self.replace(w=-self.w)

    def __call__(self, x):
        return jnp.dot(
            self.w,
            vmap(type(self.kernel).__call__, (None, None, 0))(self.kernel, x, self.X),
        )

    def rkhs_norm(self, squared=False) -> Array:
        K_xx = self.kernel.make_distance_matrix(self.X, self.X)
        norm_squared = jnp.dot(self.w, jnp.dot(K_xx, self.w))
        return jax.lax.cond(
            squared, lambda _: norm_squared, lambda _: jnp.sqrt(norm_squared), None
        )

    def grad(self, x: Array) -> Array:
        # Small helper function to be able to access velocity fields at inference time by
        # reconstructing them from their respective witness functions
        # Storing velocity fields directly would not have been easy because grad(self.__call__)
        # would yield a non-pytree object. In theory, even more structure could be added
        # since this velocity field is actually RKHS element in H^d.
        return grad(self.__call__)(x)

class DiscreteProbability(struct.PyTreeNode, metaclass=abc.ABCMeta):
    #from https://github.com/hudsonchen/DrMMD/blob/main/kwgflows/divergences/mmd.py
    X: Array
    w: Array

    @property
    def num_atoms(self):
        return self.X.shape[0]

    @classmethod
    def from_samples(cls: Type[Self], X: Array):
        assert X.ndim == 2
        num_samples = X.shape[0]
        return cls(X, jnp.ones(num_samples) / num_samples)

    def average_of(self, f: Callable[[Array], Array]) -> Array:
        return jnp.average(vmap(f)(self.X), weights=self.w)

    def push_forward(self, f: Callable[[Array], Array]) -> Self:
        new_Xs = vmap(f)(self.X)
        return self.replace(X=new_Xs)

    def get_mean_embedding(self, kernel: base_kernel) -> rkhs_element:
        return rkhs_element(self.X, self.w, kernel)

class Divergence(struct.PyTreeNode, metaclass=abc.ABCMeta):
    #from https://github.com/hudsonchen/DrMMD/blob/main/kwgflows/divergences/mmd.py
    @abc.abstractmethod
    def __call__(self, p: DiscreteProbability) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def get_first_variation(self, p: DiscreteProbability) -> Callable[[Array], Array]:
        raise NotImplementedError


class KernelizedDivergence(Divergence):
    #from https://github.com/hudsonchen/DrMMD/blob/main/kwgflows/divergences/mmd.py
    @abc.abstractmethod
    def get_first_variation(self, p: DiscreteProbability) -> rkhs_element:
        raise NotImplementedError

class GradientFlowResult(struct.PyTreeNode):
    #from https://github.com/hudsonchen/DrMMD/blob/main/kwgflows/divergences/mmd.py
    divergence: KernelizedDivergence
    Ys: DiscreteProbability

    def get_Yt(self, t):
        return tree_map(lambda x: x[t, :], self.Ys)

    def get_Y_all(self):
        return tree_map(lambda x: x, self.Ys)
    """
    def get_velocity_field(self, t, reverse=False):
        Yt = self.get_Yt(t)
        first_variation = self.divergence.get_first_variation(Yt)
        first_variation = jax.lax.cond(
            reverse, lambda _: -first_variation, lambda _: first_variation, None
        )
        return grad(first_variation)"""

class Learner(struct.PyTreeNode):
    #general EBM base class used for KSD methods

    @abc.abstractmethod
    def sample(self, n: int, theta: Array, rng_key: Array) -> Array:
        #generate random sample from the distribution
        raise NotImplementedError

    @abc.abstractmethod
    def log_prob(self, theta: Array, x: Array) -> Scalar:
        #return the log-probability of x given the parameter value is theta
        raise NotImplementedError

    def score(self, theta: Array, x: Array) -> Array:
        #use autodiff to calculate gradient of log-probability w.r.t. data point x
        dx1 = grad(self.log_prob,argnums=1)(theta,x)
        return dx1

