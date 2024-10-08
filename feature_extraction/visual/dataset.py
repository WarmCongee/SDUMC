# *_*coding:utf-8 *_*
import os
import glob
from PIL import Image
from skimage import io
import torch.utils.data as data


class FaceDataset(data.Dataset):
    def __init__(self, vid, face_dir, transform=None):
        super(FaceDataset, self).__init__()
        self.vid = vid
        self.path = os.path.join(face_dir, vid)
        self.transform = transform
        self.frames = self.get_frames()

    def get_frames(self):
        frames = glob.glob(os.path.join(self.path, '*'))
        return frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        path = self.frames[index]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        name = os.path.basename(path)[:-4]
        return img, name


class MixupFaceDataset(data.Dataset):
    def __init__(self, vid, face_dir, mix_dic, transform=None):
        super(MixupFaceDataset, self).__init__()
        self.vids = mix_dic[vid]['name']
        self.weights = mix_dic[vid]['weight']
        self.face_dir = face_dir
        # self.path = os.path.join(face_dir, vid)
        self.transform = transform
        self.frames = self.get_frames()

    def get_frames(self):
        frames = glob.glob(os.path.join(self.face_dir, self.vids[0],'*'))
        return frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        path = self.frames[index]
        # print(path)
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        img = self.weights[0] * img
        name = os.path.basename(path)[:-4]
        
        for i in range(1, len(self.weights)):
            sub_path = path.replace(self.vids[0], self.vids[i])
            if os.path.exists(sub_path):
                # print(sub_path)
                sub_img = Image.open(sub_path)
                sub_img = self.transform(sub_img)
                img = img + self.weights[i]*sub_img
                
        return img, name


class FaceDatasetForEmoNet(data.Dataset):
    def __init__(self, vid, face_dir, transform=None, augmentor=None):
        super(FaceDatasetForEmoNet, self).__init__()
        self.vid = vid
        self.path = os.path.join(face_dir, vid)
        self.augmentor = augmentor
        self.transform = transform
        self.frames = self.get_frames()

    def get_frames(self):
        frames = glob.glob(os.path.join(self.path, '*'))
        return frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        path = self.frames[index]
        img = io.imread(path)
        if self.augmentor is not None:
            img = self.augmentor(img)[0]
        if self.transform is not None:
            img = self.transform(img)
        name = os.path.basename(path)[:-4]
        return img, name