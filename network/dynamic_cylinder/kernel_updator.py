import torch.nn as nn


class KernelUpdator(nn.Module):
    def __init__(self, feature_channels=128, kernel_channels=128, out_channels=128):
        super(KernelUpdator, self).__init__()
        self.feature_channels = feature_channels
        self.kernel_channels = kernel_channels
        self.out_channels = out_channels
        self.num_params_in = self.feature_channels
        self.num_params_out = self.feature_channels
        self.kernel_conv_layer = nn.Linear(kernel_channels, self.num_params_in + self.num_params_out)
        self.feature_conv_layer = nn.Linear(feature_channels, self.num_params_in + self.num_params_out)

        self.feature_gate = nn.Linear(self.num_params_in, self.num_params_in)
        # self.feature_norm = nn.LayerNorm(self.num_params_in)
        self.feature_norm = nn.BatchNorm1d(self.num_params_in)
        self.weights_gate = nn.Linear(self.num_params_in, self.num_params_in)
        # self.weights_norm = nn.LayerNorm(self.num_params_in)
        self.weights_norm = nn.BatchNorm1d(self.num_params_in)

        # self.feature_norm_out = nn.LayerNorm(self.num_params_out)
        # self.weights_norm_out = nn.LayerNorm(self.num_params_out)
        self.feature_norm_out = nn.BatchNorm1d(self.num_params_out)
        self.weights_norm_out = nn.BatchNorm1d(self.num_params_out)

        self.fc_layer = nn.Linear(self.num_params_out, self.out_channels)
        # self.fc_norm = nn.LayerNorm(self.out_channels)
        self.fc_norm = nn.BatchNorm1d(self.out_channels)

        self.activation = nn.ReLU()

        # self.weight_initialization()

    # def weight_initialization(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.BatchNorm1d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    def forward(self, kernel_feature, kernel_weights):
        """
        @param kernel_feature: [cls, feature_channels]
        @param kernel_weights: [cls, kernel_channels], cls is the numer of classes
        """
        parameters = self.kernel_conv_layer(kernel_weights)  # [cls, num_params_in+num_params_out]
        param_in = parameters[:, :self.num_params_in].view(-1, self.num_params_in)
        param_out = parameters[:, -self.num_params_out:].view(-1, self.num_params_out)

        features = self.feature_conv_layer(kernel_feature)  # [cls, num_params_in+num_params_out]
        features_in = features[:, :self.num_params_in].view(-1, self.num_params_in)
        features_out = features_in[:, -self.num_params_out:].view(-1, self.num_params_out)

        gate_feats = param_in * features_in  # [cls, num_params_in]
        kernel_feature_gate = self.feature_norm(self.feature_gate(gate_feats)).sigmoid()
        kernel_weights_gate = self.weights_norm(self.weights_gate(gate_feats)).sigmoid()

        features_out = self.feature_norm_out(features_out)
        param_out = self.weights_norm_out(param_out)

        # [cls, channels]
        features = kernel_feature_gate * features_out + kernel_weights_gate * param_out
        features = self.activation(self.fc_norm(self.fc_layer(features)))

        return features  # [cls, out_channels]

