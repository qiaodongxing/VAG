import os
from argparse import ArgumentParser
from re import split
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import imageio
import jax
from dataset_2 import *
import cv2
import skimage



# def get_freer_gpu():
#     os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
#     memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
#     return np.argmax(memory_available)


# Import jax only after setting the visible gpu
import jax
import jax.numpy as jnp
import plenoxel
from jax.ops import index, index_update, index_add
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)


data_dir = "/mnt/2t/qjh_medical/NSCLC_DIF_Processed/attenuation/LUNG1-420/all.pickle"
np.random.seed(2024)
gpu = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
print(f'gpu is {gpu}')


def total_variation_regularization(image):
    """
    计算体素数据的Total Variation正则化项
    """
    dx = jnp.abs(image[:, :-1, :] - image[:, 1:, :])
    dy = jnp.abs(image[:-1, :, :] - image[1:, :, :])
    dz = jnp.abs(image[:, :, :-1] - image[:, :, 1:])
    tv = jnp.sum(dx) + jnp.sum(dy) + jnp.sum(dz)
    # print('tv:',tv)
    return tv


def get_loss_rays(data_dict, rays, gt, resolution, radius, uniform, penalty, interpolation,):
    acc, weights, voxel_ids,error_one_ray = plenoxel.render_rays(data_dict, rays, resolution, radius, uniform, interpolation)
    mse = jnp.mean((acc - gt[:,0].squeeze())**2)

    indices, data = data_dict
    loss_trim =  jnp.mean(jax.nn.relu(data[-1] - 1.0)) + jnp.mean(jax.nn.relu(-data[-1]))
    loss_l1 =  jnp.mean(jax.nn.relu(data[-1]))
    loss_tv = total_variation_regularization(data[-1].reshape((resolution,resolution,resolution)))

    loss = mse+ penalty * loss_l1+ penalty * loss_trim  + 3e-9*penalty*loss_tv #3.7e-12     1.23e-3
    return loss


def render_view_rays(data_dict,view_o,view_d, H, W,  resolution, radius,  uniform, batch_size, interpolation):
    accs = []
    for i in range(int(np.ceil(H*W/batch_size))):
        start = i*batch_size
        stop = min(H*W, (i+1)*batch_size)
        acc, weights, voxel_ids,error_one_ray= jax.lax.stop_gradient(plenoxel.render_rays(data_dict, (view_o[start:stop], view_d[start:stop]),
                                                                                            resolution,  radius,uniform, interpolation))
        accs.append(acc)
    accs = jnp.reshape(jnp.concatenate(accs, axis=0), (H, W))
    return accs


def test_step(i,test_dataset,data_dict, FLAGS,  name_appendage=''):
    print(f'precomputing all the eval rays')
    # Precompute all the training rays and shuffle them
    rays,test_gt = test_dataset.get_test_rays() # [N, ro+rd, H, W, 3] 50 2 256 256 3
    n_test_imgs,H, W = test_gt.shape[0], test_gt.shape[1], test_gt.shape[2]
    tpsnr = 0.0
    psnr_2d = []
    ssim_2d = []
    mse_2d = []
    all_test_img = []
    for j in tqdm.tqdm(range(len(rays))):
        gt = test_gt[j,:,:,0]
        view_o, view_d = rays[j]
        view_o = np.reshape(view_o, [-1,3])
        view_d = np.reshape(view_d, [-1,3])
        acc =  render_view_rays(data_dict,view_o,view_d, H, W,FLAGS.resolution, radius,FLAGS.uniform,FLAGS.batch_size, FLAGS.interpolation)
        mse = jnp.mean((acc - gt)**2)
        psnr = -10.0 * np.log(mse) / np.log(10.0)
        tpsnr += psnr
        psnr_value = skimage.metrics.peak_signal_noise_ratio(jax.device_get(acc), gt.squeeze(), data_range=1)
        psnr_2d.append(psnr_value)
 
        ssim =  skimage.metrics.structural_similarity(jax.device_get(acc), gt.squeeze(), multichannel=False,data_range=1)
        ssim_2d.append(ssim)
        mse_2d.append(skimage.metrics.mean_squared_error(jax.device_get(acc), gt.squeeze()))
        
        if FLAGS.render_interval > 0 and j % FLAGS.render_interval == 0:
            normal_gt = (gt.squeeze() - gt.squeeze().min()) / (gt.squeeze().max() - gt.squeeze().min())
            normal_acc = (acc - acc.min()) / (acc.max() - acc.min())
            vis = jnp.concatenate((normal_gt, normal_acc), axis=1)
            vis = np.asarray((vis * 255)).astype(np.uint8)
            vis_acc = np.asarray((normal_acc * 255)).astype(np.uint8)
            imageio.imwrite(f"{log_dir}/test_image/pred_{j:04}_{i:04}{name_appendage}.png", vis_acc)
            gt = np.asarray((normal_gt.squeeze() * 255)).astype(np.uint8)
            imageio.imwrite(f"{log_dir}/test_image/gt_{j:04}_{i:04}{name_appendage}.png", gt)
            
            vis = jnp.concatenate((gt.squeeze(), acc), axis=1)

            #归一化vis
            img = (acc-vis.min())/(vis.max()-vis.min())
            all_test_img.append(img)

            dif = np.abs(gt.squeeze() - acc)
            # 显示热力图，使用 'coolwarm' 色图
            plt.imshow(dif, cmap='coolwarm', interpolation='nearest')
            plt.colorbar()
            plt.savefig(f"{log_dir}/test_image/{j:04}_{i:04}{name_appendage}_heatmap.png")
            plt.close()  
        del acc
    
    tpsnr /= n_test_imgs
    img_3d = np.stack(all_test_img,axis=0)
    save_img_path = f"{log_dir}/test_image/test_img.npy"
    np.save(save_img_path,img_3d)
    print(f"psnr_2d:{np.mean(psnr_2d)}")
    print(f"ssim_2d:{np.mean(ssim_2d)}")
    print(f"mse_2d:{np.mean(mse_2d)}")
    return tpsnr

def update_grids(old_grid, lrs, grid_grad):
    for i in range(len(old_grid)):
        old_grid[i] = index_add(old_grid[i], index[...], -1 * lrs[i] * grid_grad[i])
    return old_grid

def normal_image(image):
    image = (image - image.min()) / (image.max() - image.min())
    image = np.asarray((image * 255)).astype(np.uint8)
    return image

def show_slice(gt_3d,rec_3d,epoch):
    rec_3d = np.maximum(0, rec_3d)
    rec_3d = normal_image(rec_3d)

    show_slice_num = 10
    show_step = gt_3d.shape[-1]//show_slice_num
    # print('show_step:',show_step)
    show_image = gt_3d[...,::show_step]
    show_image_pred = rec_3d[...,::show_step]
    show = []
    for i_show in range(show_slice_num):
        gt_normal = normal_image(show_image[..., i_show])

        out = show_image_pred[..., i_show]
        pred_normal = normal_image(out)

        save_pred_path = f"{log_dir}/slices/{epoch}/pred/pred_{i_show}.png"
        os.makedirs(os.path.dirname(save_pred_path), exist_ok=True)
        cv2.imwrite(save_pred_path, pred_normal)

        save_gt_path = f"{log_dir}/slices/{epoch}/gt/gt_{i_show}.png"
        os.makedirs(os.path.dirname(save_gt_path), exist_ok=True)
        imageio.imwrite(save_gt_path, gt_normal)
        
        show.append(np.concatenate([gt_normal,pred_normal], axis=0))
    
    show_density = np.concatenate(show, axis=1)
    imageio.imwrite(f"{log_dir}/show_density_{epoch}.png", show_density)
    print(f"save image {log_dir}/show_density_{epoch}.png")

   
def main():
    global rays_rgb, penalty, data_dict, FLAGS, radius, automatic_lr
    start_epoch = 0

    if FLAGS.reload_epoch is not None:
        start_epoch = FLAGS.reload_epoch + 1
    penalty = FLAGS.penalty / (len(rays_rgb) // FLAGS.batch_size) # 0.0012345  0.1/(3276800//4000)
    print(f'occupancy penalty is {penalty}')
    for i in range(start_epoch, FLAGS.num_epochs):
       
        # Shuffle rays over all training images
        rays_rgb = rays_rgb.take(np.random.permutation(rays_rgb.shape[0]), axis=0) # 打乱数据 3276800 3 3
        print('epoch', i)
        indices, data = data_dict
        
        for k in tqdm.tqdm(range(len(rays_rgb) // FLAGS.batch_size)):
            logical_grad = None

            effective_j = k
            batch = rays_rgb[effective_j*FLAGS.batch_size:(effective_j+1)*FLAGS.batch_size] # [B, 2+1, 3*?]
            batch_rays, target_s = (batch[:,0,:], batch[:,1,:]), batch[:,2,:]
            mse, data_grad = jax.value_and_grad(lambda grid: get_loss_rays((indices, grid), batch_rays, target_s, FLAGS.resolution, radius, 
                                                                FLAGS.uniform, penalty, FLAGS.interpolation))(data) 
        
            lrs = [FLAGS.lr_sigma]
            data = update_grids(data, lrs, data_grad)
            del data_grad, logical_grad
        data_dict = (indices, data)
        del indices, data
     
        if i % FLAGS.save_interval == FLAGS.save_interval - 1 or i == FLAGS.num_epochs - 1:
            print(f'Saving checkpoint at epoch {i}')
            plenoxel.save_grid(data_dict, os.path.join(log_dir, f'epoch_{i}'))
            gt_3d = train_dataset.get_3d_image()
            np.save(os.path.join(log_dir, f'epoch_{i}/gt_3d.npy'),gt_3d)

        #  test step    if i == FLAGS.num_epochs - 1:
        if i % FLAGS.val_interval == 0 or i == FLAGS.num_epochs - 1:
            # validation_psnr = run_test_step(i + 1, data_dict, test_c2w, test_gt, H, W, focal, FLAGS, render_keys)
            gt_3d = train_dataset.get_3d_image()
            indices, data = data_dict
            rec_3d = data[-1].reshape(gt_3d.shape)
            show_slice(gt_3d,rec_3d,i)

        if i == FLAGS.num_epochs - 1:
            validation_psnr = test_step(i ,test_dataset,data_dict, FLAGS, name_appendage='_final')
            print(f'at epoch {i}, test psnr is {validation_psnr}')


if __name__ == "__main__":
    # gpu = get_freer_gpu()

    flags = ArgumentParser()
    flags.add_argument( "--expname", type=str, default='lung_420_pure',help="Experiment name.")
    flags.add_argument( "--log_dir", type=str, default='jax_logs/',help="Directory to save outputs.")
    flags.add_argument("--resolution",  type=int,  default=256,help="Grid size.")
    flags.add_argument( "--ini_attenuation", type=float, default=0.0,help="Initial harmonics value in grid.")
    flags.add_argument( "--radius", type=float, default=0.128, help="Grid radius is 0.128 ,which is matcthing the grid size 0.256")
    flags.add_argument( '--num_epochs',  type=int, default=10, help='Epochs to train for.' )
    flags.add_argument( '--render_interval', type=int,  default=40, help='Render images during test/val step every x images.')
    flags.add_argument( '--val_interval', type=int, default=1, help='Run test/val step every x epochs.')
    flags.add_argument( '--batch_size',type=int,default=4000, help='Number of rays per batch, to avoid OOM.')
    flags.add_argument('--reload_epoch',type=int,default=None, help='Epoch at which to resume training from a saved model.')
    flags.add_argument('--save_interval',type=int,default=5, help='Save the grid checkpoints after every x epochs.')

    flags.add_argument('--interpolation', type=str, default='trilinear',
        help='Type of interpolation to use. Options are constant, trilinear, or tricubic.')
    flags.add_argument( '--lr_sigma', type=float, default=None,
        help='SGD step size for sigma. Default chooses automatically based on resolution.')
    flags.add_argument('--uniform', type=float, default=0.5,
        help='Initialize sample locations to be uniformly spaced at this interval (as a fraction of voxel_len), \
        rather than at voxel intersections (default if uniform=0).')
    flags.add_argument('--penalty',type=float,default=0.1,
        help='Penalty in the loss term for occupancy; encourages a sparse grid.')

    FLAGS = flags.parse_args()
    radius = FLAGS.radius

    train_dataset = TIGREDataset(data_dir, n_rays=1024, type="train", device="cpu")
    test_dataset = TIGREDataset(data_dir, n_rays=1024, type="val", device="cpu")

    log_dir = FLAGS.log_dir + FLAGS.expname
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir+'/test_image', exist_ok=True)
    os.makedirs(log_dir+'/slices', exist_ok=True)

    # 检查是否需要自动设置学习率
    automatic_lr = FLAGS.lr_sigma is None
    if automatic_lr:
        FLAGS.lr_sigma = 51.5 * (FLAGS.resolution ** 2.37)   #2626492.8
    
    if FLAGS.reload_epoch is not None:
        reload_dir = os.path.join(log_dir, f'epoch_{FLAGS.reload_epoch}')
        print(f'Reloading the grid from {reload_dir}')
        data_dict = plenoxel.load_grid(dirname=reload_dir, sh_dim = (FLAGS.harmonic_degree + 1)**2)
    else:
        print(f'Initializing the grid')
        data_dict = plenoxel.initialize_grid(resolution=FLAGS.resolution, ini_sigma=FLAGS.ini_attenuation)

    print(f'precomputing all the training rays')
    rays = train_dataset.get_train_rays() # [N, ro+rd, H, W, 3] 50 2 256 256 3
    rgb = train_dataset.get_train_rgb() # [N, H, W, 3] 50 256 256 3
    rays_rgb = np.concatenate([rays, rgb[:,None].astype(np.float32)], 1) # [N, ro+rd+rgb, H, W,   3] 50 3 256 256 3
    rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
    rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
    rays_rgb = rays_rgb.take(np.random.permutation(rays_rgb.shape[0]), axis=0)

    main()
