import torch
import pickle
import os
import sys
import numpy as np
import tqdm
import cv2

from torch.utils.data import DataLoader, Dataset


class ConeGeometry(object):
    """
    Cone beam CT geometry. Note that we convert to meter from millimeter.
    """
    def __init__(self, data):

        # VARIABLE                                          DESCRIPTION                    UNITS
        # -------------------------------------------------------------------------------------
        self.DSD = data["DSD"]/1000 # Distance Source Detector      (m)
        self.DSO = data["DSO"]/1000  # Distance Source Origin        (m)
        # Detector parameters
        self.nDetector = np.array(data["nDetector"])  # number of pixels              (px)
        self.dDetector = np.array(data["dDetector"])/1000  # size of each pixel    0.02        (m)
        self.sDetector = self.nDetector * self.dDetector  # total size of the detector    (m)
        # Image parameters
        self.nVoxel = np.array(data["nVoxel"])  # number of voxels              (vx)
        self.dVoxel = np.array(data["dVoxel"])/1000  # size of each voxel            (m)
        self.sVoxel = self.nVoxel * self.dVoxel  # total size of the image       (m)

        # Offsets
        self.offOrigin = np.array(data["offOrigin"])/1000  # Offset of image from origin   (m)
        self.offDetector = np.array(data["offDetector"])/1000  # Offset of Detector            (m)

        # Auxiliary
        self.accuracy = data["accuracy"]  # Accuracy of FWD proj          (vx/sample)  # noqa: E501
        # Mode
        self.mode = data["mode"]  # parallel, cone                ...
        self.filter = data["filter"]


class TIGREDataset(Dataset):
    """
    TIGRE dataset.
    """
    def __init__(self, path, n_rays=1024, type="train", device="cuda"):    
        super().__init__()

        with open(path, "rb") as handle:
            data = pickle.load(handle)
        self.data = data
        self.geo = ConeGeometry(data)
        self.type = type
        self.n_rays = n_rays
        self.near, self.far = self.get_near_far(self.geo)#
    
        if type == "train":
            self.projs = torch.tensor(data["train"]["projections"], dtype=torch.float32, device=device) #50 256 256
            self.angles = data["train"]["angles"]
            self.image = torch.tensor(data["image"], dtype=torch.float32, device=device)
        elif type == "val":
            self.projs = torch.tensor(data["val"]["projections"], dtype=torch.float32, device=device)
            self.angles = data["val"]["angles"]
            self.image = torch.tensor(data["image"], dtype=torch.float32, device=device)

            
    def get_train_rays(self):
        angeles = self.angles
        idx = np.linspace(0, len(angeles)-1, 50, dtype=np.int32)
        angeles = angeles[idx]
        print('selected angles:',idx )
        rays = self.get_rays(angeles, self.geo,'cpu')
        rays = rays.cpu().detach().numpy()
        return rays
    
    def get_train_rgb(self):
        rgb = self.projs 
        idx = np.linspace(0, rgb.shape[0]-1, 50, dtype=np.int32)
        rgb = rgb[idx]
        print('len view: ',len(idx),'selected rgb:',idx)
        rgb = rgb.unsqueeze(-1)
        rgb = rgb.repeat(1,1,1,3)
        rgb = rgb.cpu().detach().numpy()
        return rgb
    
    def get_test_rays(self):
        angeles = self.angles
        idx = np.linspace(0, len(angeles)-1, 41, dtype=np.int32)
        angeles = angeles[idx]
        print('test selected angles:',idx )
        rays = self.get_rays(angeles, self.geo,'cpu')
        rays = rays.cpu().detach().numpy()

        rgb = self.projs [idx]
        rgb = rgb.unsqueeze(-1)
        rgb = rgb.repeat(1,1,1,3)
        rgb = rgb.cpu().detach().numpy()

        return rays,rgb
    
    def get_3d_image(self):
        image = self.image
        image = image.cpu().detach().numpy()
        return image

    def get_voxels(self, geo: ConeGeometry):
        """
        Get the voxels.
        """
        n1, n2, n3 = geo.nVoxel 
        s1, s2, s3 = geo.sVoxel / 2 - geo.dVoxel / 2

        xyz = np.meshgrid(np.linspace(-s1, s1, n1),
                        np.linspace(-s2, s2, n2),
                        np.linspace(-s3, s3, n3), indexing="ij")
        voxel = np.asarray(xyz).transpose([1, 2, 3, 0])
        return voxel
    
    def get_rays(self, angles, geo: ConeGeometry, device):
        """
        Get rays given one angle and x-ray machine geometry.
        """
        W, H = geo.nDetector
        DSD = geo.DSD
        rays = []
        
        for angle in angles:
            pose = torch.Tensor(self.angle2pose(geo.DSO, angle)).to(device) #就是C2w矩阵    
            rays_o, rays_d = None, None
            if geo.mode == "cone":
                i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device),
                                    torch.linspace(0, H - 1, H, device=device))  # pytorch"s meshgrid has indexing="ij"
                uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + geo.offDetector[0]
                vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + geo.offDetector[1]
                dirs = torch.stack([uu / DSD, vv / DSD, torch.ones_like(uu)], -1)
                rays_d = torch.sum(torch.matmul(pose[:3,:3], dirs[..., None]).to(device), -1) # pose[:3, :3] * 
                rays_o = pose[:3, -1].expand(rays_d.shape)
                rays_angle = torch.stack([rays_o, rays_d], dim=0)
            
            elif geo.mode == "parallel":
                pass
                # i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device),
                #                         torch.linspace(0, H - 1, H, device=device))  # pytorch"s meshgrid has indexing="ij"
                # uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + geo.offDetector[0]
                # vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + geo.offDetector[1]
                # dirs = torch.stack([torch.zeros_like(uu), torch.zeros_like(uu), torch.ones_like(uu)], -1)
                # rays_d = torch.sum(torch.matmul(pose[:3,:3], dirs[..., None]).to(device), -1) # pose[:3, :3] * 
                # rays_o = torch.sum(torch.matmul(pose[:3,:3], torch.stack([uu,vv,torch.zeros_like(uu)],-1)[..., None]).to(device), -1) + pose[:3, -1].expand(rays_d.shape)

            else:
                raise NotImplementedError("Unknown CT scanner type!")
            rays.append(rays_angle)
        return torch.stack(rays, dim=0)

    def angle2pose(self, DSO, angle):
        phi1 = -np.pi / 2
        R1 = np.array([[1.0, 0.0, 0.0],
                    [0.0, np.cos(phi1), -np.sin(phi1)],
                    [0.0, np.sin(phi1), np.cos(phi1)]])
        phi2 = np.pi / 2
        R2 = np.array([[np.cos(phi2), -np.sin(phi2), 0.0],
                    [np.sin(phi2), np.cos(phi2), 0.0],
                    [0.0, 0.0, 1.0]])
        R3 = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0]])
        rot = np.dot(np.dot(R3, R2), R1)
        trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), 0])
        T = np.eye(4)
        T[:-1, :-1] = rot
        T[:-1, -1] = trans
        return T

    def get_near_far(self, geo: ConeGeometry, tolerance=0.005):
        """
        Compute the near and far threshold.
        """
        dist1 = np.linalg.norm([geo.offOrigin[0] - geo.sVoxel[0] / 2, geo.offOrigin[1] - geo.sVoxel[1] / 2])
        dist2 = np.linalg.norm([geo.offOrigin[0] - geo.sVoxel[0] / 2, geo.offOrigin[1] + geo.sVoxel[1] / 2])
        dist3 = np.linalg.norm([geo.offOrigin[0] + geo.sVoxel[0] / 2, geo.offOrigin[1] - geo.sVoxel[1] / 2])
        dist4 = np.linalg.norm([geo.offOrigin[0] + geo.sVoxel[0] / 2, geo.offOrigin[1] + geo.sVoxel[1] / 2])
        dist_max = np.max([dist1, dist2, dist3, dist4])
        near = np.max([0, geo.DSO - dist_max - tolerance])
        far = np.min([geo.DSO * 2, geo.DSO + dist_max + tolerance])
        return near, far
    

if __name__ == "__main__":
    # Test the dataset.

    path = "/mnt/2t/qjh_medical/NSCLC_DIF_Processed/attenuation/LUNG1-420/all.pickle"
    dataset = TIGREDataset(path, n_rays=1024, type="train", device="cpu")
    rays = dataset.get_train_rays()
    print(rays.shape)

