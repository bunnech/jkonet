#!/usr/bin/python3
# author: Charlotte Bunne

# imports
import jax.numpy as jnp
from ott.core import sinkhorn
from ott.geometry import pointcloud


def sinkhorn_loss(x, y, epsilon=0.1, div=False):
    """Computes transport between (x, a) and (y, b) via Sinkhorn algorithm."""
    a = jnp.ones(len(x)) / len(x)
    b = jnp.ones(len(y)) / len(y)

    # compute cost
    geom_xy = pointcloud.PointCloud(x, y, epsilon=epsilon)

    # solve ot problem
    out_xy = sinkhorn.sinkhorn(geom_xy, a, b)

    # compute additional terms of Sinkhorn divergence
    if div:
        geom_xx = pointcloud.PointCloud(x, x, epsilon=epsilon)
        geom_yy = pointcloud.PointCloud(y, y, epsilon=epsilon)

        out_xx = sinkhorn.sinkhorn(geom_xx, a, a)
        out_yy = sinkhorn.sinkhorn(geom_yy, b, b)

        return (
          out_xy.reg_ot_cost - 0.5 * (out_xx.reg_ot_cost + out_yy.reg_ot_cost)
          + 0.5 * geom_xy.epsilon * (jnp.sum(a) - jnp.sum(b))**2)

    # return regularized ot cost
    return out_xy.reg_ot_cost
