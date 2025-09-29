import torch
import torchvision.transforms.functional as FT


class live_transform():
    """
    Args:
        clip  : list of (num_frame) np.array [H, W, C] (BGR order, 0..1)
        boxes : list of (num_frame) list of (num_box, in ucf101-24 = 1) np.array [(x, y, w, h)] relative coordinate
    
    Return:
        clip  : torch.tensor [C, num_frame, H, W] (RGB order, 0..1)
        boxes : not change
    """

    def __init__(self, img_size):
        self.img_size = img_size
        # cache mean/std once
        self._mean = torch.FloatTensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        self._std  = torch.FloatTensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    def to_tensor(self, image):
        return FT.to_tensor(image)
    
    def normalize(self, clip, mean=[0.4345, 0.4051, 0.3775], std=[0.2768, 0.2713, 0.2737]):
        
        mean_tensor = torch.FloatTensor(mean).view(-1, 1, 1).to(clip.device)
        std_tensor = torch.FloatTensor(std).view(-1, 1, 1).to(clip.device)
        
        clip -= mean_tensor
        clip /= std_tensor
        return clip
    
    def __call__(self, img):
        W, H = img.size
        img = img.resize([self.img_size, self.img_size])
        img = self.to_tensor(img)
        img = self.normalize(img)
        return img
    