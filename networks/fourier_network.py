import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FourierNet(nn.Module):
    def __init__(self, input_size_hw, input_channels, crop_fraction, mlp_hidden_dims, output_dim):
        super(FourierNet, self).__init__()
        self.input_h, self.input_w = input_size_hw
        self.input_channels = input_channels
        self.crop_fraction = crop_fraction
        if self.input_channels > 1:
            print("Warning: FourierNet processes only the first channel of multi-channel images for FFT.")
        center_h, center_w = self.input_h // 2, self.input_w // 2
        _desired_crop_h = math.ceil(self.input_h * self.crop_fraction)
        _desired_crop_w = math.ceil(self.input_w * self.crop_fraction)
        if _desired_crop_h % 2 == 0 and _desired_crop_h < self.input_h: _desired_crop_h += 1
        if _desired_crop_w % 2 == 0 and _desired_crop_w < self.input_w: _desired_crop_w += 1
        self.crop_h = min(_desired_crop_h, self.input_h);
        self.crop_h = max(1, self.crop_h)
        self.crop_w = min(_desired_crop_w, self.input_w);
        self.crop_w = max(1, self.crop_w)
        self.flattened_spectrum_dim = self.crop_h * self.crop_w
        self.mlp_layers = nn.ModuleList()
        current_dim = self.flattened_spectrum_dim
        for h_dim in mlp_hidden_dims:
            self.mlp_layers.append(nn.Linear(current_dim, h_dim))
            current_dim = h_dim
        self.output_layer = nn.Linear(current_dim, output_dim)

    def _extract_fourier_features(self, x):
        x_fft = x[:, 0:1, :, :] if x.shape[1] > 1 else x
        fft_coeffs = torch.fft.fft2(x_fft)
        power_spectrum = torch.abs(fft_coeffs) ** 2
        shifted_spectrum = torch.fft.fftshift(power_spectrum, dim=(-2, -1))
        center_h, center_w = self.input_h // 2, self.input_w // 2
        h_start = center_h - self.crop_h // 2;
        h_end = h_start + self.crop_h
        w_start = center_w - self.crop_w // 2;
        w_end = w_start + self.crop_w
        h_start_clamped = max(0, h_start);
        h_end_clamped = min(self.input_h, h_end)
        w_start_clamped = max(0, w_start);
        w_end_clamped = min(self.input_w, w_end)
        cropped_spectrum = shifted_spectrum[:, :, h_start_clamped:h_end_clamped, w_start_clamped:w_end_clamped]
        flattened_features = cropped_spectrum.reshape(cropped_spectrum.size(0), -1)
        return flattened_features, cropped_spectrum, shifted_spectrum

    def forward(self, x, return_spectrum_features=False, return_penultimate_features=False):
        fourier_features, cropped_spectrum_for_viz, full_shifted_spectrum_for_viz = self._extract_fourier_features(x)
        penultimate_activation = fourier_features
        for i, layer in enumerate(self.mlp_layers):
            penultimate_activation = layer(penultimate_activation)
            penultimate_activation = F.relu(penultimate_activation)
        logits = self.output_layer(penultimate_activation)

        optional_returns = {}
        if return_spectrum_features:  # For plot_fourier_network_analysis
            optional_returns['spectrum_features'] = (cropped_spectrum_for_viz.detach().cpu(),
                                                     full_shifted_spectrum_for_viz.detach().cpu())
        if return_penultimate_features:  # For t-SNE
            optional_returns['penultimate_features'] = penultimate_activation

        if optional_returns:
            return logits, optional_returns
        else:
            return logits