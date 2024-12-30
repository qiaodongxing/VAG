import cv2
import numpy as np
import sklearn.metrics as metrics
import os
import skimage
import imageio


log_root = f'jax_logs/lung_420_pure'
f = open(os.path.join(log_root,'slices/psnr_ssim.txt'),'w')

def clal_2d_psnr_ssim(log_root):
    for j in range(10):
        pred_dir = os.path.join(log_root, 'slices/{}/pred'.format(j))
        gt_dir = os.path.join(log_root, 'slices/{}/gt'.format(j))
        gt_files = os.listdir(gt_dir)
        gt_files.sort()
        pred_files = os.listdir(pred_dir)
        pred_files.sort()
        psnrs = []
        ssims = []
        for i in range(len(gt_files)):
            if i==0 :
                continue
            gt_files[i] = os.path.join(gt_dir, gt_files[i])
            pred_files[i] = os.path.join(pred_dir, pred_files[i])
            gt = cv2.imread(gt_files[i], cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_files[i], cv2.IMREAD_GRAYSCALE)
            psnr = skimage.metrics.peak_signal_noise_ratio(gt, pred, data_range=255)
            ssim = skimage.metrics.structural_similarity(gt, pred, data_range=255)
            psnrs.append(psnr)
            ssims.append(ssim)
        print('epoch: ',j, 'psnr:', np.mean(psnrs), 'ssim:',np.mean(ssims))
        f.write('epoch: {}, psnr: {}, ssim: {} \n'.format(j, np.mean(psnrs), np.mean(ssims)))

def show_all_slices(gt_3d,rec_3d,epoch):
    rec_3d = np.maximum(0, rec_3d)
    rec_3d = normal_image(rec_3d)

    show_slice_num = 256
    show_step = gt_3d.shape[-1]//show_slice_num

    show_image = gt_3d[...,::show_step]
    show_image_pred = rec_3d[...,::show_step]
    show = []
    for i_show in range(show_slice_num):
        gt_normal = normal_image(show_image[..., i_show])
        out = show_image_pred[..., i_show]
        pred_normal = normal_image(out)

        save_pred_path = f"{log_root}/slices/{epoch}/all_pred/pred_{i_show}.png"
        os.makedirs(os.path.dirname(save_pred_path), exist_ok=True)
        cv2.imwrite(save_pred_path, pred_normal)

        save_gt_path = f"{log_root}/slices/{epoch}/all_gt/gt_{i_show}.png"
        os.makedirs(os.path.dirname(save_gt_path), exist_ok=True)
        imageio.imwrite(save_gt_path, gt_normal)
        show.append(np.concatenate([gt_normal,pred_normal], axis=0))
    show_density = np.concatenate(show, axis=1)

    imageio.imwrite(f"{log_root}/show_density_{epoch}.png", show_density)
    print(f"save image {log_root}/show_density_{epoch}.png")

def normal_image(image):
    image = (image - image.min()) / (image.max() - image.min())
    image = np.asarray((image * 255)).astype(np.uint8)
    return image  

def show_slice(gt_3d,rec_3d,epoch):
    rec_3d = np.maximum(0, rec_3d)
    rec_3d = normal_image(rec_3d)

    show_slice_num = 10
    show_step = gt_3d.shape[-1]//show_slice_num

    show_image = gt_3d[...,::show_step]
    show_image_pred = rec_3d[...,::show_step]
    show = []
    for i_show in range(show_slice_num):
        gt_normal = normal_image(show_image[..., i_show])
        out = show_image_pred[..., i_show]

        pred_normal = normal_image(out)

        save_pred_path = f"{log_root}/slices/{epoch}/pred/pred_{i_show}.png"
        os.makedirs(os.path.dirname(save_pred_path), exist_ok=True)
        cv2.imwrite(save_pred_path, pred_normal)

        save_gt_path = f"{log_root}/slices/{epoch}/gt/gt_{i_show}.png"
        os.makedirs(os.path.dirname(save_gt_path), exist_ok=True)
        imageio.imwrite(save_gt_path, gt_normal)
        show.append(np.concatenate([gt_normal,pred_normal], axis=0))
    show_density = np.concatenate(show, axis=1)

    imageio.imwrite(f"{log_root}/show_density_{epoch}.png", show_density)
    print(f"save image {log_root}/show_density_{epoch}.png")


gt_3d = np.load(f"{log_root}/epoch_9/gt_3d.npy")
rec_3d = np.load(f"{log_root}/epoch_9/sigma_grid.npy")
rec_3d = rec_3d.reshape(gt_3d.shape)

show_slice(gt_3d,rec_3d,9)
clal_2d_psnr_ssim(log_root)

rec_3d = np.maximum(0, rec_3d)
rec_3d = normal_image(rec_3d)
gt_3d = normal_image(gt_3d)
psnr_3d = skimage.metrics.peak_signal_noise_ratio(rec_3d, gt_3d, data_range=255)
ssim_3d = skimage.metrics.structural_similarity(rec_3d, gt_3d, data_range=255)
print('psnr_3d:', psnr_3d, 'ssim_3d:', ssim_3d)
f.write('psnr_3d: {}, ssim_3d: {} \n'.format(psnr_3d, ssim_3d))

psnrs = []
ssims = []
for i in range(rec_3d.shape[0]):

    pred_2d = rec_3d[i]
    gt_2d = gt_3d[i]
    psnr = skimage.metrics.peak_signal_noise_ratio(gt_2d, pred_2d)
    ssim = skimage.metrics.structural_similarity(gt_2d, pred_2d)
    psnrs.append(psnr)
    ssims.append(ssim)
print('2d psnr:', np.mean(psnrs))
print('2d ssim:', np.mean(ssims))
f.write('9 epoch 2d psnr: {}, 2d ssim: {} \n'.format(np.mean(psnrs), np.mean(ssims)))
f.close()   

