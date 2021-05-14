
# Adapted from monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/networks/pose_decoder.py

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from collections import OrderedDict


class PoseResDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseResDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))
        
        self.found_mat_net = nn.Sequential(nn.Linear(9, 18), nn.ReLU(inplace=True), 
                                           nn.Linear(18, 18), nn.ReLU(inplace=True),
                                           nn.Linear(18, 6))
            
        self.fusion_net = nn.Sequential(nn.Linear(12, 24), nn.ReLU(inplace=True),
                                        nn.Linear(24, 24), nn.ReLU(inplace=True),
                                        nn.Linear(24, 6))   

    def forward(self, input_features, foud_mat):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)
        out = out.mean(3).mean(2)

        # out = 0.01 * (out.view(-1, self.num_frames_to_predict_for, 1, 6) + self.found_mat_net(foud_mat.view(-1, 9)).view(-1, 1, 1, 6))
        
        fund_mat_proj = self.found_mat_net(foud_mat.view(-1, 9))
        out = 0.01 * (self.fusion_net(torch.cat([out.view(-1, 6), fund_mat_proj], dim=1)) + fund_mat_proj).view(-1, 1, 1, 6)

        print("out", out.view(-1, 6))
        print("fund", fund_mat_proj.view(-1, 6))  
        
        axisangle = out[..., :3]
        translation = out[..., 3:]
        return axisangle, translation



class PoseResAngleDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseResAngleDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 7 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))
        
        self.found_mat_net = nn.Sequential(nn.Linear(6, 18), nn.ReLU(inplace=True), 
                                           nn.Linear(18, 18), nn.ReLU(inplace=True),
                                           nn.Linear(18, 6))
            
        self.fusion_net = nn.Sequential(nn.Linear(12, 128), nn.ReLU(inplace=True),
                                        nn.Linear(128, 128), nn.ReLU(inplace=True),
                                        nn.Linear(128, 6))   

    def forward(self, input_features, pose_geo):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)
        out = out.mean(3).mean(2).view(-1, 7)

        # out = 0.01 * (out.view(-1, self.num_frames_to_predict_for, 1, 6) + self.found_mat_net(foud_mat.view(-1, 9)).view(-1, 1, 1, 6))
        
        #trans_scale = 0.01 * pose_geo[:, 3:] * out[:, -1].unsqueeze(1)
        trans_scale = pose_geo[:, 3:] * out[:, -1].unsqueeze(1)

        pose_geo_new = torch.cat([pose_geo[:, :3], trans_scale], dim=1)
        
        #out = 0.01 * (self.fusion_net(torch.cat([out[:, :6], self.found_mat_net(pose_geo_new)], dim=1)))
        # out = 0.01 * (self.fusion_net(torch.cat([0.01 * out[:, :6], pose_geo_new], dim=1)))
        out = 0.01 * (self.fusion_net(torch.cat([out[:, :6], pose_geo_new], dim=1)))

        # out = 0.01 * out[:, :6] + pose_geo_new
        
        out = out.view(-1, 1, 1, 6)
        
        axisangle = out[..., :3]
        translation = out[..., 3:]
        return axisangle, translation

