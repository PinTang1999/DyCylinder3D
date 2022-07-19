import torch
import torch.nn as nn
import torch.nn.functional as F
from network.dynamic_cylinder.kernel_updator import KernelUpdator


class PointSegmentor(nn.Module):
    def __init__(self, input_point_dim=256, input_voxel_dim=256, middle_dim=256, num_class=20):
        super(PointSegmentor, self).__init__()

        self.seg_model = nn.Sequential(
            nn.BatchNorm1d(input_point_dim + input_voxel_dim),
            nn.Linear(input_point_dim + input_voxel_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.cls_seg = nn.Linear(64, num_class)

    def forward(self, point_feature, voxel_feature, p2vmap, voxel_logits):
        """
        :param point_feature, the point feature before voxelization, [n_points, C1]
        :param voxel_feature, the voxel feature outputted by sparse U-net, [n_voxel, C2]
        :param p2vmap, the mapping from point idx to voxel idx, [n_point]
        :param voxel_logits, the logits of the voxels corresponding to class, [n_voxel, n_class]
        """
        unet_point_feature = voxel_feature[p2vmap]
        concated_point_feature = torch.concat((point_feature, unet_point_feature), 1)
        seg_feature = self.seg_model(concated_point_feature)

        seg_logits = self.cls_seg(seg_feature)

        return seg_logits


class DynamicPointSegmentor(nn.Module):
    def __init__(self, input_point_dim=256, input_voxel_dim=256, middle_dim=256, num_class=20):
        super(DynamicPointSegmentor, self).__init__()
        self.linear1 = nn.Linear(input_point_dim + input_voxel_dim, middle_dim)
        self.bn1 = nn.BatchNorm1d(middle_dim)
        self.act1 = nn.ReLU()

        self.linear2 = nn.Linear(input_voxel_dim, middle_dim)
        self.bn2 = nn.BatchNorm1d(middle_dim)
        # self.act2 = nn.ReLU()

    def forward(self, point_feature, voxel_feature, p2vmap, voxel_logits):
        unet_point_feature = voxel_feature[p2vmap]  # [n_point, C2]
        concated_point_feature = torch.concat((point_feature, unet_point_feature), 1)  # [n_points, C1+C2]
        concated_point_feature = self.act1(self.bn1(self.linear1(concated_point_feature)))  # [n_points, C]

        # get the cls-wise feature
        cls_prob = F.softmax(voxel_logits, dim=1)  # [n_voxel, n_class]
        cls_fea = torch.einsum('bc,bf->cf', cls_prob, voxel_feature)  # [n_class, C2]
        # cls_fea = self.act2(self.bn2(self.linear2(cls_fea)))  # [n_class, C]
        # TODO: the bn2 may need to be moved
        cls_fea = self.bn2(self.linear2(cls_fea))  # [n_class, C]

        # use the dynamic weights on point features
        seg_logits = F.linear(concated_point_feature, cls_fea)  # [n_points, n_class]

        return seg_logits


class DynamicSegmentor(nn.Module):
    def __init__(self,
                 feature_channels=128,
                 kernel_channels=128,
                 out_channels=128,
                 num_head=8):
        super(DynamicSegmentor, self).__init__()
        self.kernel_update_conv = KernelUpdator(feature_channels=feature_channels,
                                                kernel_channels=kernel_channels,
                                                out_channels=out_channels)
        self.out_dim = out_channels

        # self.linear_q = nn.Linear(self.out_dim, self.out_dim)
        # self.linear_k = nn.Linear(self.out_dim, self.out_dim)
        # self.linear_v = nn.Linear(self.out_dim, self.out_dim)
        # self.attention = nn.MultiheadAttention(self.out_dim, num_head)
        # self.attention_norm = nn.LayerNorm(self.out_dim)
        # self.attention_norm = nn.BatchNorm1d(self.out_dim)

        self.ffn = nn.Linear(self.out_dim, self.out_dim)
        # self.ffn_norm = nn.LayerNorm(self.out_dim)
        self.ffn_norm = nn.BatchNorm1d(self.out_dim)

        self.reg_layer = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            # nn.LayerNorm(self.out_dim),
            nn.BatchNorm1d(self.out_dim),
            nn.ReLU()
        )

        self.fc_mask = nn.Linear(self.out_dim, self.out_dim)

        # self.weight_initialization()

    # def weight_initialization(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.BatchNorm1d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    def forward(self, voxel_feature, voxel_logits, kernel_weights):
        """
        @param voxel_feature: [M, C]
        @param voxel_logits: [M, cls]
        @param kernel_weights: [cls, C]
        """
        voxel_logits = voxel_logits.softmax(dim=1)
        kernel_feature = torch.einsum('bc,bf->cf', voxel_logits, voxel_feature)  # [cls, C]
        # num_class, kernel_channels = kernel_weights.shape[-2:]
        # # [k, k, k, in_channel, cls]->[k*k*k, cls, in_channels]->[cls, k^3, in_channels]
        # kernel_weights = kernel_weights.reshape(-1, kernel_channels, num_class).permute(1, 0, 2)
        updated_kernel = self.kernel_update_conv(kernel_feature, kernel_weights)  # [cls, C]

        # updated_kernel = updated_kernel.reshape(num_class, -1)  # [cls, k^3*out_channels]

        # q = self.linear_q(updated_kernel).unsqueeze(1)
        # k = self.linear_k(updated_kernel).unsqueeze(1)
        # v = self.linear_v(updated_kernel).unsqueeze(1)
        #
        # updated_kernel = self.attention_norm(self.attention(q, k, v)[0].squeeze())  # [cls, k^3*out_channels]
        new_kernel = self.ffn(updated_kernel) + updated_kernel
        new_kernel = self.ffn_norm(new_kernel)

        new_kernel = self.reg_layer(new_kernel)  # [cls, out_channels]

        new_kernel = self.fc_mask(new_kernel)

        new_voxel_logits = F.linear(voxel_feature, new_kernel)  # [M, cls]

        return new_voxel_logits, new_kernel

