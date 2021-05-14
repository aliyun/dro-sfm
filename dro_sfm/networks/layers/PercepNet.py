import torch
import torch.nn as nn
import torchvision


class PercepNet(torch.nn.Module):
    def __init__(self, requires_grad=False, resize=True):
        super(PercepNet, self).__init__()
        mean_rgb = torch.FloatTensor([0.485, 0.456, 0.406])
        std_rgb = torch.FloatTensor([0.229, 0.224, 0.225])
        self.register_buffer('mean_rgb', mean_rgb)
        self.register_buffer('std_rgb', std_rgb)
        self.resize = resize

        vgg_pretrained_features = torchvision.models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def normalize(self, x):
        out = (x - self.mean_rgb.view(1,3,1,1)) / self.std_rgb.view(1,3,1,1)
        if self.resize:
            out = nn.functional.interpolate(out, mode='bilinear',size=(224, 224), align_corners=False)
        return out
    
    def forward(self, im1, im2):
        im = torch.cat([im1,im2], 0)
        im = self.normalize(im)  # normalize input

        ## compute features
        feats = []
        f = self.slice1(im)
        h, w = f.shape[-2:]
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice2(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice3(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice4(f)
        feats += [torch.chunk(f, 2, dim=0)]

        losses = []
        #weights = [0.3, 0.3, 0.4]  
        #weights = [0.15, 0.25, 0.25, 0.25]  
        weights = [0.15, 0.25, 0.6]  
        #weights = [1.0, 1.0, 1.0]  
        for i, (f1, f2) in enumerate(feats[0:3]):  
            loss = weights[i] * torch.abs(f1-f2).mean(1, True) #(B, 1, H, W)
            loss = nn.functional.interpolate(loss, mode='bilinear',size=(h, w), align_corners=False)
            losses += [loss]
        
        return sum(losses)
    
