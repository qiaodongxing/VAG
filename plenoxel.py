import jax
import jax.numpy as jnp
from jax import lax
from jax.ops import index, index_update
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from jax import random

# the rendering function based on the naf code
def my_rending( sigma, z_vals, dirs, raw_noise_std = 0.0):
  eps = 1e-10
  dists = z_vals[Ellipsis, 1:] - z_vals[Ellipsis, :-1]
  dists = dists * jnp.linalg.norm(dirs[Ellipsis, None, :], axis=-1)  # Convert ray-relative distance to absolute distance (shouldn't matter if rays_d is normalized)
  noise = 0.
  if raw_noise_std > 0.:
    key = random.PRNGKey(0)
    noise = random.normal(key, sigma.shape) * raw_noise_std
  acc = jnp.sum((jax.nn.relu(sigma) + noise) * dists, axis=-1)
  weights = 0.
  # if sigma.shape[-1] == 1:
  #   eps = jnp.ones_like(sigma[:, :1, -1]) * 1e-10
  #   weights = jnp.concatenate([eps, jnp.abs(sigma[:, 1:, -1] - sigma[:, :-1, -1])], axis=-1)
  #   weights = weights / jnp.max(weights)
  # elif sigma.shape[-1] == 2: # with jac
  #   weights = sigma[..., 1] / jnp.max(sigma[..., 1])
  # else:
    # raise NotImplementedError("Wrong raw shape")
  return acc, weights


eps = 1e-5

def near_zero(vector):
  return jnp.abs(vector) < eps


def safe_floor(vector):
  return jnp.floor(vector + eps)


def safe_ceil(vector):
  return jnp.ceil(vector - eps)


@jax.partial(jax.jit, static_argnums=(2,3,4,5))
def intersection_distances(inputs, data_dict, resolution, radius,  uniform,  interpolation):
  start, stop, offset, interval = inputs["start"], inputs["stop"], inputs["offset"], inputs["interval"]
  if uniform == 0:
    # For a single ray, compute all the possible voxel intersections up to the upper bound number, starting when the ray hits the cube
    upper_bound = int(1 + resolution) # per dimension upper bound on the number of voxel intersections
    intersections0 = jnp.linspace(start=start[0] + offset[0], stop=start[0] + offset[0] + interval[0] * upper_bound, num=upper_bound, endpoint=False)
    intersections1 = jnp.linspace(start=start[1] + offset[1], stop=start[1] + offset[1] + interval[1] * upper_bound, num=upper_bound, endpoint=False)
    intersections2 = jnp.linspace(start=start[2] + offset[2], stop=start[2] + offset[2] + interval[2] * upper_bound, num=upper_bound, endpoint=False)
    intersections = jnp.concatenate([intersections0, intersections1, intersections2], axis=None)
    intersections = jnp.sort(intersections) # TODO: replace this with just a merge of the three intersection arrays
  else:
    voxel_len = radius * 2.0 / resolution
    realstart = jnp.min(start)
    count = int(resolution*3 / uniform)
    intersections = jnp.linspace(start=realstart + uniform*voxel_len, stop=realstart + uniform*voxel_len*(count+1), num=count, endpoint=False)
  intersections = jnp.where(intersections <= stop, intersections, stop)
  # Get the values at these intersection points
  ray_o, ray_d = inputs["ray_o"], inputs["ray_d"]
  voxel_sigma, intersections,error_one_ray = values_oneray(intersections, data_dict, ray_o, ray_d, resolution, radius,interpolation, 1e-5)
  return voxel_sigma, intersections,error_one_ray


get_intersections_partial = jax.vmap(fun=intersection_distances, in_axes=({"start": 0, "stop": 0, "offset": 0, "interval": 0, "ray_o": 0, "ray_d": 0}, None, None, None, None, None, ), out_axes=0)
get_intersections = jax.vmap(fun=get_intersections_partial, in_axes=({"start": 1, "stop": 1, "offset": 1, "interval": 1, "ray_o": 1, "ray_d": 1}, None, None, None, None, None,), out_axes=1)


@jax.partial(jax.jit, static_argnums=(3,4))
def voxel_ids_oneray(intersections, ray_o, ray_d, voxel_len, resolution, eps=1e-5):
  # For a single ray, compute the ids of all the voxels it passes through
  # Compute the midpoint of the ray segment inside each voxel
  midpoints = (intersections[Ellipsis, 1:] + intersections[Ellipsis, :-1]) / 2.0
  midpoints = ray_o[jnp.newaxis, :] + midpoints[:, jnp.newaxis] * ray_d[jnp.newaxis, :]
  ids = jnp.array(jnp.floor(midpoints / voxel_len + eps) + resolution / 2, dtype=int)
  return ids

voxel_ids_partial = jax.jit(jax.vmap(fun=voxel_ids_oneray, in_axes=(0, 0, 0, None, None), out_axes=0))
voxel_ids = jax.jit(jax.vmap(fun=voxel_ids_partial, in_axes=(1, 1, 1, None, None), out_axes=1))


def scalarize(i, j, k, resolution):
  return i*resolution*resolution + j*resolution + k


def vectorize(index, resolution):
  i = index // (resolution**2)
  j = (index - i*resolution*resolution) // resolution
  k = index - i*resolution*resolution - j*resolution
  return jnp.array([i, j, k])


def initialize_grid(resolution, ini_sigma=0.1,):
  data = []  # data is a list of length sh_dim + 1
  data.append(jnp.ones((resolution**3), dtype=np.float32) * ini_sigma) #data has 9 sh and 1 sigma each sh has 3 channels  
  indices = jnp.arange(resolution**3, dtype=int).reshape((resolution, resolution, resolution))
  return (indices, data)


def save_grid(grid, dirname):
  indices, data = grid
  if not os.path.exists(dirname):
    os.makedirs(dirname)
  sigma = data[-1]
  relu_sigma = jnp.clip(sigma, a_min=0, a_max=1e7)
  np.save(os.path.join(dirname, f'sigma_grid.npy'), relu_sigma)
  np.save(os.path.join(dirname, f'indices.npy'), indices)


def load_grid(dirname, sh_dim):
  data = []
  for i in range(sh_dim):
    data.append(np.load(os.path.join(dirname, f'sh_grid{i}.npy')))
  data.append(np.load(os.path.join(dirname, f'sigma_grid.npy')))
  indices = np.load(os.path.join(dirname, f'indices.npy'))
  return (indices, data)

  
@jax.jit
def trilinear_interpolation_weight(xyzs):
  # xyzs should have shape [n_pts, 3] and denote the offset (as a fraction of voxel_len) from the 000 interpolation point
  xs = xyzs[:,0]
  ys = xyzs[:,1]
  zs = xyzs[:,2]
  weight000 = (1-xs) * (1-ys) * (1-zs)  # [n_pts]
  weight001 = (1-xs) * (1-ys) * zs  # [n_pts]
  weight010 = (1-xs) * ys * (1-zs)  # [n_pts]
  weight011 = (1-xs) * ys * zs  # [n_pts]
  weight100 = xs * (1-ys) * (1-zs)  # [n_pts]
  weight101 = xs * (1-ys) * zs  # [n_pts]
  weight110 = xs * ys * (1-zs)  # [n_pts]
  weight111 = xs * ys * zs  # [n_pts]
  weights =  jnp.stack([weight000, weight001, weight010, weight011, weight100, weight101, weight110, weight111], axis=-1) # [n_pts, 8]
  return weights

def apply_power(power, xyzs):
  return xyzs[:,0]**power[0] * xyzs[:,1]**power[1] * xyzs[:,2]**power[2]

@jax.jit
def grid_lookup(x, y, z, grid):
  indices, data = grid
  ret = [jnp.where(indices[x,y,z,jnp.newaxis]>=0, d[indices[x,y,z]], jnp.zeros(3)) for d in data[:-1]]
  ret.append(jnp.where(indices[x,y,z]>=0, data[-1][indices[x,y,z]], 0))
  return ret


@jax.partial(jax.jit, static_argnums=(4,5,6))
def values_oneray(intersections, grid, ray_o, ray_d, resolution, radius,interpolation, eps=1e-5):
  voxel_len = radius * 2.0 / resolution
  pts = ray_o[jnp.newaxis, :] + intersections[:, jnp.newaxis] * ray_d[jnp.newaxis, :]  # [n_intersections, 3] #o+td 1 3 + 1536 1* 1 3 = 1536 3
  pts = pts[:, jnp.newaxis, :]  # [n_intersections, 1, 3]
  offsets = jnp.array([[-1,-1,-1], [-1,-1,1], [-1,1,-1], [-1,1,1], [1,-1,-1], [1,-1,1], [1,1,-1], [1,1,1]]) * voxel_len / 2.0  # [8, 3]
  neighbors = jnp.clip(pts + offsets[jnp.newaxis, :, :], a_min=-radius, a_max=radius)  # [n_intersections, 8, 3]
  neighbor_centers = jnp.clip((jnp.floor(neighbors / voxel_len + eps) + 0.5) * voxel_len, a_min=-(radius - voxel_len/2), a_max=radius - voxel_len/2)  # [n_intersections, 8, 3]
  neighbor_ids = jnp.array(jnp.floor(neighbor_centers / voxel_len + eps) + resolution / 2, dtype=int)  # [n_intersections, 8, 3] 将在中心的下标转化成左下角的下标
  neighbor_ids = jnp.clip(neighbor_ids, a_min=0, a_max=resolution-1) #1536 8 3
  xyzs = (pts[:,0,:] - neighbor_centers[:,0,:]) / voxel_len # [n_intersections, 3]

  if interpolation == 'trilinear':
    weights = trilinear_interpolation_weight(xyzs)  # [n_intersections, 8]
    neighbor_data = grid_lookup(neighbor_ids[...,0], neighbor_ids[...,1], neighbor_ids[...,2], grid)
    # neighbor_sh = neighbor_data[:-1]
    neighbor_sigma = neighbor_data[-1]
    error_one_ray=0.0
    pt_sigma = jnp.sum(weights * neighbor_sigma, axis=1)[:-1]
    # pt_sh = [jnp.sum(weights[..., jnp.newaxis] * nsh, axis=1)[:-1,:] for nsh in neighbor_sh]
  elif interpolation == 'constant':
    voxel_ids = neighbor_ids[:,0,:]
    voxel_data = jax.vmap(lambda voxel_id: grid_lookup(voxel_id[0], voxel_id[1], voxel_id[2], grid))(voxel_ids)
    pt_sigma = voxel_data[-1][:-1]
    
  else:
    print(f'Unrecognized interpolation method {interpolation}.')
    assert False
  return  pt_sigma, intersections,error_one_ray
  

@jax.partial(jax.jit, static_argnums=(2,3,4,5))
def render_rays(grid, rays, resolution,radius=1.3, uniform=0, interpolation='trilinear'):

  voxel_len = radius * 2.0 / resolution #0.001 0.128*2/256
  assert (resolution // 2) * 2 == resolution # Renderer assumes resolution is a multiple of 2
  assert voxel_len==0.001
  rays_o, rays_d = rays
  # Compute when the rays enter and leave the grid
  offsets_pos = jax.lax.stop_gradient((radius - rays_o) / rays_d)  # r = o+td -> t = (r-o)/d  xyz方向与边界的交点的t
  offsets_neg = jax.lax.stop_gradient((-radius - rays_o) / rays_d)
  offsets_in = jax.lax.stop_gradient(jnp.minimum(offsets_pos, offsets_neg))
  offsets_out = jax.lax.stop_gradient(jnp.maximum(offsets_pos, offsets_neg))
  start = jax.lax.stop_gradient(jnp.max(offsets_in, axis=-1, keepdims=True)) #最开始的时间
  stop = jax.lax.stop_gradient(jnp.min(offsets_out, axis=-1, keepdims=True))
  first_intersection = jax.lax.stop_gradient(rays_o + start * rays_d)
  # Compute locations of ray-voxel intersections along each dimension
  interval = jax.lax.stop_gradient(voxel_len / jnp.abs(rays_d)) #detal t  计算了光线穿过一个体素所需的时间 interval
  offset_bigger = jax.lax.stop_gradient((safe_ceil(first_intersection / voxel_len) * voxel_len - first_intersection) / rays_d)
  offset_smaller = jax.lax.stop_gradient((safe_floor(first_intersection / voxel_len) * voxel_len - first_intersection) / rays_d)
  offset = jax.lax.stop_gradient(jnp.maximum(offset_bigger, offset_smaller))

  # Compute the samples along each ray
  voxel_sigma, intersections,error_one_ray = get_intersections_partial({"start": start, "stop": stop, "offset": offset, "interval": interval, "ray_o": rays_o, "ray_d": rays_d}, 
                                                                     grid, resolution, radius,  uniform,  interpolation)

  # Call volumetric_rendering
  acc,weights = my_rending(voxel_sigma, intersections, rays_d)
 
  pts = rays_o[:, jnp.newaxis, :] + intersections[:, :, jnp.newaxis] * rays_d[:, jnp.newaxis, :]  # [n_rays, n_intersections, 3]
  ids = jnp.clip(jnp.array(jnp.floor(pts / voxel_len + eps) + resolution / 2, dtype=int), a_min=0, a_max=resolution-1)
  return  acc, weights, ids,error_one_ray

# def get_rays(H, W, focal, c2w):
#   i, j = jnp.meshgrid(jnp.linspace(0, W-1, W) + 0.5, jnp.linspace(0, H-1, H) + 0.5) 
#   dirs = jnp.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -jnp.ones_like(i)], -1)
#   # Rotate ray directions from camera frame to the world frame
#   rays_d = jnp.sum(dirs[..., jnp.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
#   # Translate camera frame's origin to the world frame. It is the origin of all rays.
#   rays_o = jnp.broadcast_to(c2w[:3,-1], rays_d.shape)
#   return rays_o, rays_d 

