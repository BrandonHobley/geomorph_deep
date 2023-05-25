from torch.utils.data.dataset import Dataset
import scipy.io as sio
import torch
import os


# DATALOADER CLASS TO READ THE DATASET GENERATED FROM THE PRE-PROCESS MATLAB CODE
class GeomorphIR(Dataset):

    def __init__(self, root, split, transform=None, target_transform=None):

        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.samples = []
        self.labels = []

        self.img_file_dir = os.path.join(self.root, self.split, 'imgs')
        self.lab_file_dir = os.path.join(self.root, self.split, 'labels')

        imgs_list = os.listdir(self.img_file_dir)
        lab_list = os.listdir(self.lab_file_dir)

        for img_file in imgs_list:
            img_id = ''.join([i for i in img_file if i.isdigit()])
            self.samples.append((img_file, img_id))

        for lab_file in lab_list:
            lab_id = ''.join([i for i in lab_file if i.isdigit()])
            self.labels.append((lab_file, lab_id))

    def __getitem__(self, index):

        img, _ = self.samples[index]
        label, _ = self.labels[index]

        img_path = os.path.join(self.img_file_dir, img)
        label_path = os.path.join(self.lab_file_dir, label)

        im = sio.loadmat(img_path)
        lab = sio.loadmat(label_path)

        # KEY DEPENDANT ON MATLAB PRE-PROCESS CODE
        t_im = im['t_img']
        t_lab = torch.from_numpy(lab['l_img'])

        if self.transform is not None:
            t_im = self.transform(t_im)

        t_im = t_im.type(torch.FloatTensor)

        t_lab = t_lab.long()

        return t_im, t_lab

    def __len__(self):
        return len(self.samples)
