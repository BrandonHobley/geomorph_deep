from torch.utils.data.dataset import Dataset
import scipy.io as sio
import re
import os


class GeomorphIR(Dataset):
    def __init__(self, root, transform=None, target_transform=None):

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.samples = []

        imgs_list = os.listdir(self.root)
        for img_file in imgs_list:
            if img_file.endswith('.mat'):
                coords = list(map(int, re.findall(r'\d+', img_file)))
                self.samples.append((img_file, coords))

    def __getitem__(self, index):

        img, coords = self.samples[index]

        img_path = os.path.join(self.root, img)

        im = sio.loadmat(img_path)
        t_im = im['img']

        if self.transform is not None:
            t_im = self.transform(t_im)

        return t_im, img_path

    def __len__(self):
        return len(self.samples)
