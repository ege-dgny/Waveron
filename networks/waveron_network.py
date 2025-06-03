import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
import math


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class Waveron(nn.Module):
    def __init__(self, in_channels, conv_out_channels, kernel_size, wavelet_name, J=1):
        super(Waveron, self).__init__()
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.kernel_size = kernel_size

        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, conv_out_channels,
                              kernel_size=kernel_size, padding=padding, bias=True)
        self.bn = nn.BatchNorm2d(conv_out_channels)
        self.relu = nn.ReLU()
        self.dwt = DWTForward(J=J, wave=wavelet_name, mode='symmetric')
        
        # Add residual connection if input and output channels match
        self.use_residual = in_channels == conv_out_channels
        if self.use_residual:
            self.residual_conv = nn.Conv2d(in_channels, conv_out_channels, kernel_size=1)
            
        # Add channel attention
        self.channel_attention = ChannelAttention(conv_out_channels)

    def forward(self, x):
        identity = x
        
        convolved = self.conv(x)
        normalized = self.bn(convolved)
        activated = self.relu(normalized)
        
        # Apply channel attention
        attention = self.channel_attention(activated)
        activated = activated * attention
        
        # Add residual connection if applicable
        if self.use_residual:
            activated = activated + self.residual_conv(identity)
            
        Yl, Yh = self.dwt(activated)
        Yh0 = Yh[0]
        LH = Yh0[:, :, 0, :, :]
        HL = Yh0[:, :, 1, :, :]
        HH = Yh0[:, :, 2, :, :]
        return Yl, LH, HL, HH


class WaveNetClassifierBase(nn.Module):
    def __init__(self, num_classes=10, wavelet_name='haar', kernel_size=3,
                 waveron_conv_channels=1,
                 input_size_hw=(28, 28), input_channels=1,
                 aggregation_type='sum', num_waveron_layers=2):
        super(WaveNetClassifierBase, self).__init__()
        self.wavelet_name = wavelet_name
        self.kernel_size = kernel_size
        self.waveron_conv_channels = waveron_conv_channels
        self.input_size_h, self.input_size_w = input_size_hw
        self.input_channels = input_channels
        self.aggregation_type = aggregation_type
        self.num_waveron_layers = num_waveron_layers

        # Add learnable aggregation weights
        self.aggregation_weights = nn.Parameter(torch.ones(4) / 4)  # Initialize with equal weights
        self.aggregation_softmax = nn.Softmax(dim=0)

        self.initial_dwt = DWTForward(J=1, wave=self.wavelet_name, mode='symmetric')

        with torch.no_grad():
            _dummy_in_l0 = torch.randn(1, self.input_channels, self.input_size_h, self.input_size_w)
            _s1_yl, _ = self.initial_dwt(_dummy_in_l0)

            _dummy_waveron1 = Waveron(in_channels=self.input_channels,
                                      conv_out_channels=self.waveron_conv_channels,
                                      kernel_size=self.kernel_size,
                                      wavelet_name=self.wavelet_name)
            _s2_yl_from_waveron1, _, _, _ = _dummy_waveron1(_s1_yl)

            if self.num_waveron_layers == 2:
                if self.aggregation_type == 'sum':
                    waveron2_in_channels_dummy = self.waveron_conv_channels
                elif self.aggregation_type == 'concat':
                    waveron2_in_channels_dummy = 4 * self.waveron_conv_channels
                else:
                    raise ValueError(f"Unknown aggregation type: {self.aggregation_type}")

                _dummy_waveron2 = Waveron(in_channels=waveron2_in_channels_dummy,
                                          conv_out_channels=self.waveron_conv_channels,
                                          kernel_size=self.kernel_size,
                                          wavelet_name=self.wavelet_name)
                _dummy_input_for_w2 = torch.randn(1, waveron2_in_channels_dummy, _s2_yl_from_waveron1.shape[2],
                                                  _s2_yl_from_waveron1.shape[3])
                _s3_yl, _, _, _ = _dummy_waveron2(_dummy_input_for_w2)
                final_feature_map_h, final_feature_map_w = _s3_yl.shape[2], _s3_yl.shape[3]
                final_band_channels = self.waveron_conv_channels
                num_final_bands = 16
            elif self.num_waveron_layers == 1:
                final_feature_map_h, final_feature_map_w = _s2_yl_from_waveron1.shape[2], _s2_yl_from_waveron1.shape[3]
                final_band_channels = self.waveron_conv_channels
                num_final_bands = 16
            else:
                raise ValueError(f"Unsupported num_waveron_layers: {self.num_waveron_layers}")

        self.waverons1 = nn.ModuleList()
        for _ in range(4):
            self.waverons1.append(
                Waveron(in_channels=self.input_channels,
                        conv_out_channels=self.waveron_conv_channels,
                        kernel_size=self.kernel_size,
                        wavelet_name=self.wavelet_name)
            )

        if self.num_waveron_layers == 2:
            if self.aggregation_type == 'sum':
                self.waveron2_in_channels = self.waveron_conv_channels
            elif self.aggregation_type == 'concat':
                self.waveron2_in_channels = 4 * self.waveron_conv_channels
            self.waverons2 = nn.ModuleList()
            for _ in range(4):
                self.waverons2.append(
                    Waveron(in_channels=self.waveron2_in_channels,
                            conv_out_channels=self.waveron_conv_channels,
                            kernel_size=self.kernel_size,
                            wavelet_name=self.wavelet_name)
                )
        else:
            self.waverons2 = None

        features_per_final_band = final_band_channels * final_feature_map_h * final_feature_map_w
        self.total_fc_input_features = num_final_bands * features_per_final_band
        self.fc = nn.Linear(self.total_fc_input_features, num_classes)

    def _aggregate_w1_outputs(self, w1_outputs_by_type_dict_of_lists):
        aggregated_bands = []
        if self.aggregation_type == 'sum':
            for i in range(4):
                bands_of_one_type = w1_outputs_by_type_dict_of_lists[i]
                # Apply learnable weights to each band
                weighted_bands = [band * self.aggregation_softmax(self.aggregation_weights)[i] 
                                for band in bands_of_one_type]
                aggregated_bands.append(torch.sum(torch.stack(weighted_bands, dim=0), dim=0))
        elif self.aggregation_type == 'concat':
            for i in range(4):
                bands_of_one_type = w1_outputs_by_type_dict_of_lists[i]
                # Apply learnable weights to each band
                weighted_bands = [band * self.aggregation_softmax(self.aggregation_weights)[i] 
                                for band in bands_of_one_type]
                aggregated_bands.append(torch.cat(weighted_bands, dim=1))
        return aggregated_bands

    def forward(self, x, return_intermediate_outputs=False, return_penultimate_features=False,
                return_final_subband_maps=False):  # Added new flag
        Yl0, Yh0_list = self.initial_dwt(x)
        initial_bands_on_device = [Yl0, Yh0_list[0][:, :, 0, :, :], Yh0_list[0][:, :, 1, :, :],
                                   Yh0_list[0][:, :, 2, :, :]]

        all_w1_path_outputs_collected_viz = [] if return_intermediate_outputs else None
        all_w1_path_outputs_collected_energy = [] if return_intermediate_outputs else None
        w1_outputs_by_type_for_agg = [[] for _ in range(4)]

        for i in range(4):
            waveron1_module = self.waverons1[i]
            input_to_waveron1 = initial_bands_on_device[i]
            ll, lh, hl, hh = waveron1_module(input_to_waveron1)
            w1_outputs_by_type_for_agg[0].append(ll);
            w1_outputs_by_type_for_agg[1].append(lh)
            w1_outputs_by_type_for_agg[2].append(hl);
            w1_outputs_by_type_for_agg[3].append(hh)
            if return_intermediate_outputs:
                all_w1_path_outputs_collected_viz.append([b.detach().cpu() for b in [ll, lh, hl, hh]])
                all_w1_path_outputs_collected_energy.append([ll, lh, hl, hh])

        final_bands_before_fc = []  # This will hold the 16 bands that go to FC

        if self.num_waveron_layers == 1:
            # Collect all 16 bands from w1_outputs_by_type_for_agg directly
            for path_idx in range(4):  # For each of the 4 original paths
                final_bands_before_fc.append(w1_outputs_by_type_for_agg[0][path_idx])
                final_bands_before_fc.append(w1_outputs_by_type_for_agg[1][path_idx])
                final_bands_before_fc.append(w1_outputs_by_type_for_agg[2][path_idx])
                final_bands_before_fc.append(w1_outputs_by_type_for_agg[3][path_idx])
            current_penultimate_features = torch.cat([t.reshape(t.size(0), -1) for t in final_bands_before_fc], dim=1)

        elif self.num_waveron_layers == 2:
            aggregated_bands_for_w2 = self._aggregate_w1_outputs(w1_outputs_by_type_for_agg)
            w2_0_LLpath_outputs_viz = None
            if return_intermediate_outputs:
                ll_w2_0, lh_w2_0, hl_w2_0, hh_w2_0 = self.waverons2[0](aggregated_bands_for_w2[0])
                w2_0_LLpath_outputs_viz = [b.detach().cpu() for b in [ll_w2_0, lh_w2_0, hl_w2_0, hh_w2_0]]
            for i in range(4):
                waveron2_module = self.waverons2[i]
                input_to_waveron2 = aggregated_bands_for_w2[i]
                ll, lh, hl, hh = waveron2_module(input_to_waveron2)
                final_bands_before_fc.extend([ll, lh, hl, hh])  # Collect these 16 bands
            current_penultimate_features = torch.cat([t.reshape(t.size(0), -1) for t in final_bands_before_fc], dim=1)
        else:
            raise ValueError(f"Unsupported num_waveron_layers: {self.num_waveron_layers}")

        output_logits = self.fc(current_penultimate_features)

        optional_returns = {}
        if return_penultimate_features:
            optional_returns['penultimate_features'] = current_penultimate_features
        if return_final_subband_maps:  # New flag
            optional_returns['final_subband_maps'] = [b.detach().cpu() for b in final_bands_before_fc]

        if return_intermediate_outputs:  # For detailed feature plotting
            viz_data = {'input_image': x.detach().cpu(),
                        'initial_bands': [b.detach().cpu() for b in initial_bands_on_device],
                        'all_waveron1_path_outputs': all_w1_path_outputs_collected_viz if all_w1_path_outputs_collected_viz is not None else []}
            if self.num_waveron_layers == 2:
                viz_data['summed_bands_layer2_input'] = [b.detach().cpu() for b in aggregated_bands_for_w2]
                viz_data[
                    'waveron2_0_LLpath_outputs'] = w2_0_LLpath_outputs_viz if w2_0_LLpath_outputs_viz is not None else []
            optional_returns['viz_intermediates'] = viz_data

            energy_data = {
                'initial_bands': initial_bands_on_device,
                'all_waveron1_path_outputs': all_w1_path_outputs_collected_energy if all_w1_path_outputs_collected_energy is not None else []
            }
            # If num_waveron_layers == 1, final_bands_before_fc are the same as all_w1_path_outputs (just structured differently)
            # If num_waveron_layers == 2, final_bands_before_fc are outputs of waverons2.
            # The energy plot currently focuses on L1. This could be extended.
            optional_returns['energy_intermediates'] = energy_data

        if optional_returns:
            return output_logits, optional_returns
        else:
            return output_logits


class WaveNetClassifier(WaveNetClassifierBase):
    def __init__(self, num_classes=10, wavelet_name='haar', kernel_size=3, waveron_conv_channels=1,
                 input_size_hw=(28, 28), input_channels=1):
        super(WaveNetClassifier, self).__init__(num_classes, wavelet_name, kernel_size, waveron_conv_channels,
                                                input_size_hw, input_channels, aggregation_type='sum',
                                                num_waveron_layers=2)


class WaveNetClassifierConcat(WaveNetClassifierBase):
    def __init__(self, num_classes=10, wavelet_name='haar', kernel_size=3, waveron_conv_channels=1,
                 input_size_hw=(28, 28), input_channels=1):
        super(WaveNetClassifierConcat, self).__init__(num_classes, wavelet_name, kernel_size, waveron_conv_channels,
                                                      input_size_hw, input_channels, aggregation_type='concat',
                                                      num_waveron_layers=2)


class SingleLayerWaveronConcat(WaveNetClassifierBase):
    def __init__(self, num_classes=10, wavelet_name='haar', kernel_size=3, waveron_conv_channels=1,
                 input_size_hw=(28, 28), input_channels=1):
        super(SingleLayerWaveronConcat, self).__init__(num_classes, wavelet_name, kernel_size, waveron_conv_channels,
                                                       input_size_hw, input_channels, aggregation_type='concat',
                                                       num_waveron_layers=1)
