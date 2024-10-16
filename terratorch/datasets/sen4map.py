import numpy as np
import h5py

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from terratorch.datasets.utils import HLSBands

from torchvision.transforms.v2.functional import resize
from torchvision.transforms.v2 import InterpolationMode

import pickle


class Sen4MapDatasetMonthlyComposites(Dataset):
    #  This dictionary maps the LUCAS classes to LULC classes. This class currently does not accommodate the crop classification task of sen4map.
    d={'A10':1, 'A11':1, 'A12':1, 'A13':1, 
    'A20':1, 'A21':1, 'A30':1, 
    'A22':2, 'F10':2, 'F20':2, 
    'F30':2, 'F40':2,
    'E10':3, 'E20':3, 'E30':3, 'B50':3, 'B51':3, 'B52':3,
    'B53':3, 'B54':3, 'B55':3,
    'B10':4, 'B11':4, 'B12':4, 'B13':4, 'B14':4, 'B15':4,
    'B16':4, 'B17':4, 'B18':4, 'B19':4, 'B10':4, 'B20':4, 
    'B21':4, 'B22':4, 'B23':4, 'B30':4, 'B31':4, 'B32':4,
    'B33':4, 'B34':4, 'B35':4, 'B30':4, 'B36':4, 'B37':4,
    'B40':4, 'B41':4, 'B42':4, 'B43':4, 'B44':4, 'B45':4,
    'B70':4, 'B71':4, 'B72':4, 'B73':4, 'B74':4, 'B75':4,
    'B76':4, 'B77':4, 'B80':4, 'B81':4, 'B82':4, 'B83':4,
    'B84':4, 
    'BX1':4, 'BX2':4,
    'C10':5, 'C20':6, 'C21':6, 'C22':6,
    'C23':6, 'C30':6, 'C31':6, 'C32':6,
    'C33':6, 
    'CXX1':6, 'CXX2':6, 'CXX3':6, 'CXX4':6, 'CXX5':6,
    'CXX5':6, 'CXX6':6, 'CXX7':6, 'CXX8':6, 'CXX9':6,
    'CXXA':6, 'CXXB':6, 'CXXC':6, 'CXXD':6, 'CXXE':6,
    'D10':7, 'D20':7, 'D10':7,
    'G10':8, 'G11':8, 'G12':8, 'G20':8, 'G21':8, 'G22':8, 'G30':8, 
    'G40':8,
    'G50':8,
    'H10':9, 'H11':9, 'H12':9, 'H11':9,'H20':9, 'H21':9,
    'H22':9, 'H23':9, '': 10}
    
    def __init__(
            self,
            h5py_file_object:h5py.File,
            h5data_keys = None,
            crop_size:None|int = None,
            dataset_bands:list[HLSBands|int]|None = None,
            input_bands:list[HLSBands|int]|None = None,
            resize = False,
            resize_to = [224, 224],
            resize_interpolation = InterpolationMode.BILINEAR,
            resize_antialiasing = True,
            reverse_tile = False,
            reverse_tile_size = 3,
            save_keys_path = None
            ):
        self.h5data = h5py_file_object
        if h5data_keys is None:
            self.h5data_keys = list(self.h5data.keys())
            if save_keys_path is not None:
                with open(save_keys_path, "wb") as file:
                    pickle.dump(self.h5data_keys, file)
        else:
            self.h5data_keys = h5data_keys
        self.crop_size = crop_size
        if input_bands and not dataset_bands:
            raise ValueError(f"input_bands was provided without specifying the dataset_bands")
        # self.dataset_bands = dataset_bands
        # self.input_bands = input_bands
        if input_bands and dataset_bands:
            self.input_channels = [dataset_bands.index(band_ind) for band_ind in input_bands if band_ind in dataset_bands]
        else: self.input_channels = None

        self.resize = resize
        self.resize_to = resize_to
        self.resize_interpolation = resize_interpolation
        self.resize_antialiasing = resize_antialiasing
        
        self.reverse_tile = reverse_tile
        self.reverse_tile_size = reverse_tile_size

    def __getitem__(self, index):
        # we can call dataset with an index, eg. dataset[0]
        im = self.h5data[self.h5data_keys[index]]
        Image, Label = self.get_data(im)
        Image = self.min_max_normalize(Image, [67.0, 122.0, 93.27, 158.5, 160.77, 174.27, 162.27, 149.0, 84.5, 66.27 ],
                                    [2089.0, 2598.45, 3214.5, 3620.45, 4033.61, 4613.0, 4825.45, 4945.72, 5140.84, 4414.45])
        
        Image = Image.clip(0,1)
        Label = torch.LongTensor(Label)
        if self.input_channels:
            Image = Image[self.input_channels, ...]

        return {"image":Image, "label":Label}

    def __len__(self):
        return len(self.h5data_keys)

    def get_data(self, im):
        mask = im['SCL'] < 9

        B2= np.where(mask==1, im['B2'], 0)
        B3= np.where(mask==1, im['B3'], 0)
        B4= np.where(mask==1, im['B4'], 0)
        B5= np.where(mask==1, im['B5'], 0)
        B6= np.where(mask==1, im['B6'], 0)
        B7= np.where(mask==1, im['B7'], 0)
        B8= np.where(mask==1, im['B8'], 0)
        B8A= np.where(mask==1, im['B8A'], 0)
        B11= np.where(mask==1, im['B11'], 0)
        B12= np.where(mask==1, im['B12'], 0)
        Image = np.stack((B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12), axis=0, dtype="float32")
        Image = np.moveaxis(Image, [0],[1])
        Image = torch.from_numpy(Image)
        
        # Composites:
        n1= [i for i, s in enumerate(im.attrs['Image_ID'].tolist()) if '201801' in s]
        n2= [i for i, s in enumerate(im.attrs['Image_ID'].tolist()) if '201802' in s]
        n3= [i for i, s in enumerate(im.attrs['Image_ID'].tolist()) if '201803' in s]
        n4= [i for i, s in enumerate(im.attrs['Image_ID'].tolist()) if '201804' in s]
        n5= [i for i, s in enumerate(im.attrs['Image_ID'].tolist()) if '201805' in s]
        n6= [i for i, s in enumerate(im.attrs['Image_ID'].tolist()) if '201806' in s]
        n7= [i for i, s in enumerate(im.attrs['Image_ID'].tolist()) if '201807' in s]
        n8= [i for i, s in enumerate(im.attrs['Image_ID'].tolist()) if '201808' in s]
        n9= [i for i, s in enumerate(im.attrs['Image_ID'].tolist()) if '201809' in s]
        n10= [i for i, s in enumerate(im.attrs['Image_ID'].tolist()) if '201810' in s]
        n11= [i for i, s in enumerate(im.attrs['Image_ID'].tolist()) if '201811' in s]
        n12= [i for i, s in enumerate(im.attrs['Image_ID'].tolist()) if '201812' in s]


        Jan= n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9 + n10 + n11 + n12 if not n1 else n1
        Feb= n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9 + n10 + n11 + n12 if not n2 else n2
        Mar= n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9 + n10 + n11 + n12 if not n3 else n3
        Apr= n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9 + n10 + n11 + n12 if not n4 else n4
        May= n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9 + n10 + n11 + n12 if not n5 else n5
        Jun= n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9 + n10 + n11 + n12 if not n6 else n6
        Jul= n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9 + n10 + n11 + n12 if not n7 else n7
        Aug= n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9 + n10 + n11 + n12 if not n8 else n8
        Sep= n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9 + n10 + n11 + n12 if not n9 else n9
        Oct= n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9 + n10 + n11 + n12 if not n10 else n10
        Nov= n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9 + n10 + n11 + n12 if not n11 else n11
        Dec= n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9 + n10 + n11 + n12 if not n12 else n12

        month_indices = [Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec]

        month_medians = [torch.stack([Image[month_indices[i][j]] for j in range(len(month_indices[i]))]).median(dim=0).values for i in range(12)]
        

        Image = torch.stack(month_medians, dim=0)
        Image = torch.moveaxis(Image, 0, 1)
        
        if self.crop_size: Image = self.crop_center(Image, self.crop_size, self.crop_size)
        if self.reverse_tile:
            Image = self.reverse_tiling_pytorch(Image, kernel_size=self.reverse_tile_size)
        if self.resize:
            Image = resize(Image, size=self.resize_to, interpolation=self.resize_interpolation, antialias=self.resize_antialiasing)

        Label = im.attrs['lc1']
        Label = Sen4MapDatasetMonthlyComposites.d[Label] - 1
        Label = np.array(Label)
        Label = Label.astype('float32')

        return Image, Label
    
    def crop_center(self, img_b:torch.Tensor, cropx, cropy) -> torch.Tensor:
        c, t, y, x = img_b.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)    
        return img_b[0:c, 0:t, starty:starty+cropy, startx:startx+cropx]
    
    
    def reverse_tiling_pytorch(self, img_tensor: torch.Tensor, kernel_size: int=3):
        """
        Upscales an image where every pixel is expanded into `kernel_size`*`kernel_size` pixels.
        Used to test whether the benefit of resizing images to the pre-trained size comes from the bilnearly interpolated pixels,
        or if the same would be realized with no interpolated pixels.
        """
        assert kernel_size % 2 == 1
        assert kernel_size >= 3
        padding = (kernel_size - 1) // 2
        # img_tensor shape: (batch_size, channels, H, W)
        batch_size, channels, H, W = img_tensor.shape
        # Unfold: Extract 3x3 patches with padding of 1 to cover borders
        img_tensor = F.pad(img_tensor, pad=(padding,padding,padding,padding), mode="replicate")
        patches = F.unfold(img_tensor, kernel_size=kernel_size, padding=0)  # Shape: (batch_size, channels*9, H*W)
        # Reshape to organize the 9 values from each 3x3 neighborhood
        patches = patches.view(batch_size, channels, kernel_size*kernel_size, H, W)  # Shape: (batch_size, channels, 9, H, W)
        # Rearrange the patches into (batch_size, channels, 3, 3, H, W)
        patches = patches.view(batch_size, channels, kernel_size, kernel_size, H, W)
        # Permute to have the spatial dimensions first and unfold them
        patches = patches.permute(0, 1, 4, 2, 5, 3)  # Shape: (batch_size, channels, H, 3, W, 3)
        # Reshape to get the final expanded image of shape (batch_size, channels, H*3, W*3)
        expanded_img = patches.reshape(batch_size, channels, H * kernel_size, W * kernel_size)
        return expanded_img

    def min_max_normalize(self, tensor:torch.Tensor, q_low:list[float], q_hi:list[float]) -> torch.Tensor:
        dtype = tensor.dtype
        q_low = torch.as_tensor(q_low, dtype=dtype, device=tensor.device)
        q_hi = torch.as_tensor(q_hi, dtype=dtype, device=tensor.device)
        x = torch.tensor(-12.0)
        y = torch.exp(x)
        tensor.sub_(q_low[:, None, None, None]).div_((q_hi[:, None, None, None].sub_(q_low[:, None, None, None])).add(y))
        return tensor