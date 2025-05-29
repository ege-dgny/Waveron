import matplotlib.pyplot as plt
import numpy as np
import math
import os
import torch
from sklearn.manifold import TSNE


def tensor_to_image(tensor, channel=0, un_normalize_params=None):
    img_tensor = tensor.detach().cpu()
    if un_normalize_params:
        mean, std = un_normalize_params
        if not isinstance(mean, torch.Tensor): mean = torch.tensor(mean, device=img_tensor.device)
        if not isinstance(std, torch.Tensor): std = torch.tensor(std, device=img_tensor.device)
        if img_tensor.ndim == 3:
            if mean.ndim == 1: mean = mean.view(-1, 1, 1)
            if std.ndim == 1: std = std.view(-1, 1, 1)
        elif img_tensor.ndim == 4 and img_tensor.shape[0] == 1:
            if mean.ndim == 1: mean = mean.view(1, -1, 1, 1)
            if std.ndim == 1: std = std.view(1, -1, 1, 1)
        img_tensor = img_tensor * std + mean
        img_tensor = torch.clamp(img_tensor, 0., 1.)
    if img_tensor.ndim == 4 and img_tensor.shape[0] == 1:
        img = img_tensor.squeeze(0)
    elif img_tensor.ndim == 3:
        img = img_tensor
    elif img_tensor.ndim == 2:
        return img_tensor.numpy()
    else:
        num_elements = img_tensor.numel()
        side = int(math.sqrt(num_elements)) if num_elements > 0 else 10
        if side * side != num_elements and num_elements > 0: return np.zeros((10, 10))
        return np.zeros((side, side)) if num_elements == 0 else img_tensor.view(side, side).numpy()
    if img.ndim == 3:
        if img.shape[0] == 1:
            img_channel = img[0, :, :]
        elif img.shape[0] == 3 and channel < 3:
            img_channel = img[channel, :, :]
        elif img.shape[0] > 0:
            img_channel = img[0, :, :]
        else:
            return np.zeros((img.shape[1], img.shape[2])) if img.ndim == 3 and img.shape[1] > 0 and img.shape[
                2] > 0 else np.zeros((10, 10))
    elif img.ndim == 2:
        img_channel = img
    else:
        return np.zeros_like(img_tensor.numpy())
    return img_channel.numpy()


def plot_training_metrics(train_losses, test_losses, test_accuracies, num_epochs, save_dir):
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1);
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss');
    plt.plot(epochs, test_losses, 'ro-', label='Test Loss')
    plt.title('Training and Test Loss');
    plt.xlabel('Epochs');
    plt.ylabel('Loss');
    plt.legend();
    plt.grid(True)
    plt.subplot(1, 2, 2);
    plt.plot(epochs, test_accuracies, 'go-', label='Test Accuracy')
    plt.title('Test Accuracy');
    plt.xlabel('Epochs');
    plt.ylabel('Accuracy (%)');
    plt.legend();
    plt.grid(True)
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_filename = os.path.join(save_dir, "training_metrics.png")
        plt.savefig(plot_filename);
        print(f"Training metrics plot saved to {plot_filename}")
    plt.show(block=False)


def plot_waveron_features(model, dataloader, device, num_samples=2, save_dir=None, dataset_name_for_plot=""):
    model.eval()
    if len(dataloader) == 0: print("Dataloader empty, skipping feature plot."); return
    try:
        images_batch, labels_batch = next(iter(dataloader))
    except StopIteration:
        print("Dataloader exhausted for feature plot."); return
    sample_images = images_batch[:num_samples].to(device)
    sample_labels = labels_batch[:num_samples]
    initial_band_names_viz = ['LL0', 'LH0', 'HL0', 'HH0']
    waveron_output_band_names_viz = ['LL_out', 'LH_out', 'HL_out', 'HH_out']
    mnist_un_norm_params = ((0.1307,), (0.3081,)) if "mnist" in dataset_name_for_plot.lower() else None

    for i in range(num_samples):
        if i >= sample_images.size(0): continue
        img_tensor_for_model = sample_images[i].unsqueeze(0)
        current_label = sample_labels[i].item()

        outputs_tuple = model(img_tensor_for_model, return_intermediate_outputs=True)
        if isinstance(outputs_tuple, tuple) and len(outputs_tuple) == 2 and isinstance(outputs_tuple[1],
                                                                                       dict) and 'viz_intermediates' in \
                outputs_tuple[1]:
            viz_intermediates = outputs_tuple[1]['viz_intermediates']
        else:
            print(
                f"Could not retrieve 'viz_intermediates' for Waveron plotting from model {type(model).__name__}. Skipping feature plot.")
            return

        is_single_layer = model.num_waveron_layers == 1 if hasattr(model, 'num_waveron_layers') else False

        # Determine number of rows needed for the plot
        # Row 0 for Input, Row 1 for Initial DWT bands
        # Then 4 rows for W1 paths.
        # If 2-layer, 1 row for aggregated, 1 row for W2_0_LL_path outputs
        plot_rows = 1 + 1 + 4
        if not is_single_layer:
            plot_rows += 2

        num_cols = 4

        fig, axes = plt.subplots(plot_rows, num_cols, figsize=(num_cols * 2.8, plot_rows * 2.8),
                                 squeeze=False)  # Increased fig size slightly

        wavelet_name_for_title = model.wavelet_name if hasattr(model, 'wavelet_name') else 'N/A'
        title_detail = ""
        if hasattr(model, 'kernel_size'):
            title_detail = f"K:{model.kernel_size}, Ch_W:{model.waveron_conv_channels if hasattr(model, 'waveron_conv_channels') else 'N/A'}"

        fig.suptitle(
            f"Features Sample: {current_label} (Data: {dataset_name_for_plot}, Model: {model.__class__.__name__}, Wavelet: {wavelet_name_for_title}, InCh: {model.input_channels}, {title_detail})",
            fontsize=12, y=0.995)

        # --- Row 0: Input Image (Centered in the first row) ---
        # To center it, we can make it span columns or place it in a specific subplot and turn off others
        # For a 4-column layout, placing it in axes[0,1] and turning off 0,0 0,2 0,3 can work.
        input_img_display = tensor_to_image(viz_intermediates['input_image'], un_normalize_params=mnist_un_norm_params)
        axes[0, 1].imshow(input_img_display, cmap='gray')  # Place in a central-ish spot
        input_image_title = "Input Image";
        un_norm_suffix = " (Disp. Un-norm)" if mnist_un_norm_params else ""
        if "noisy_bg_rand" in dataset_name_for_plot.lower():
            input_image_title = f"Noisy Digit, Random BG{un_norm_suffix}"
        elif "mnist_bg_rand" in dataset_name_for_plot.lower():
            input_image_title = f"Random BG{un_norm_suffix}"
        elif "mnist_noisy" in dataset_name_for_plot.lower():
            input_image_title = f"Noisy Input Image{un_norm_suffix}"
        elif "mnist" in dataset_name_for_plot.lower() and mnist_un_norm_params:
            input_image_title = f"Input Image{un_norm_suffix}"
        axes[0, 1].set_title(input_image_title, fontsize=10);
        axes[0, 1].axis('off')
        axes[0, 0].axis('off');
        axes[0, 2].axis('off');
        axes[0, 3].axis('off')

        # --- Row 1: Initial DWT bands ---
        current_plot_row_idx = 1
        if 'initial_bands' in viz_intermediates and isinstance(viz_intermediates['initial_bands'], list) and len(
                viz_intermediates['initial_bands']) == 4:
            for k_idx, band_tensor in enumerate(viz_intermediates['initial_bands']):
                ax = axes[current_plot_row_idx, k_idx]
                ax.imshow(tensor_to_image(band_tensor, channel=0), cmap='gray');
                ax.set_title(initial_band_names_viz[k_idx], fontsize=9);
                ax.axis('off')
        else:
            print("Warning: 'initial_bands' not found or has unexpected structure. Skipping initial band plots.")
        current_plot_row_idx += 1

        # --- Rows for Waveron1 Path Outputs ---
        if 'all_waveron1_path_outputs' in viz_intermediates and isinstance(
                viz_intermediates['all_waveron1_path_outputs'], list) and len(
                viz_intermediates['all_waveron1_path_outputs']) == 4:
            for path_idx in range(4):
                path_input_name = initial_band_names_viz[path_idx]
                waveron1_path_output_bands = viz_intermediates['all_waveron1_path_outputs'][path_idx]
                if isinstance(waveron1_path_output_bands, list) and len(waveron1_path_output_bands) == 4:
                    for band_idx, band_tensor in enumerate(waveron1_path_output_bands):
                        ax = axes[current_plot_row_idx, band_idx]
                        img_to_plot = tensor_to_image(band_tensor, channel=0)
                        ax.imshow(img_to_plot, cmap='viridis');
                        ax.set_title(f"W1({path_input_name})\n->{waveron_output_band_names_viz[band_idx]}", fontsize=8);
                        ax.axis('off')
                else:
                    print(f"Warning: Waveron1 path {path_idx} output bands have unexpected structure. Skipping.")
                current_plot_row_idx += 1
        else:
            print(
                "Warning: 'all_waveron1_path_outputs' not found or has unexpected structure. Skipping L1 Waveron output plots.")

        if not is_single_layer:
            if 'summed_bands_layer2_input' in viz_intermediates and isinstance(
                    viz_intermediates['summed_bands_layer2_input'], list) and len(
                    viz_intermediates['summed_bands_layer2_input']) == 4:
                agg_band_names_viz = ['Concat_LL1', 'Concat_LH1', 'Concat_HL1',
                                      'Concat_HH1'] if 'concat' in model.__class__.__name__.lower() else ['Sum_LL1',
                                                                                                          'Sum_LH1',
                                                                                                          'Sum_HL1',
                                                                                                          'Sum_HH1']
                for k_idx, band_tensor in enumerate(viz_intermediates['summed_bands_layer2_input']):
                    ax = axes[current_plot_row_idx, k_idx];
                    img_to_plot = tensor_to_image(band_tensor, channel=0)
                    ax.imshow(img_to_plot, cmap='viridis');
                    ax.set_title(f"{agg_band_names_viz[k_idx]} (Ch0)", fontsize=8);
                    ax.axis('off')
            else:
                print("Warning: 'summed_bands_layer2_input' (aggregated) not found. Skipping for 2-layer model.")
            current_plot_row_idx += 1

            if 'waveron2_0_LLpath_outputs' in viz_intermediates and isinstance(
                    viz_intermediates['waveron2_0_LLpath_outputs'], list) and len(
                    viz_intermediates['waveron2_0_LLpath_outputs']) == 4:
                input_to_w2_0_name = "ConcatLL1" if 'concat' in model.__class__.__name__.lower() else "SumLL1"
                w2_LLpath_output_bands = viz_intermediates['waveron2_0_LLpath_outputs']
                for k_idx, band_tensor in enumerate(w2_LLpath_output_bands):
                    ax = axes[current_plot_row_idx, k_idx];
                    img_to_plot = tensor_to_image(band_tensor, channel=0)
                    ax.imshow(img_to_plot, cmap='viridis');
                    ax.set_title(f"W2({input_to_w2_0_name})\n->{waveron_output_band_names_viz[k_idx]} (Ch0)",
                                 fontsize=8);
                    ax.axis('off')
            else:
                print("Warning: 'waveron2_0_LLpath_outputs' not found. Skipping for 2-layer model.")
            current_plot_row_idx += 1

        # Turn off any remaining unused axes from the original grid if plot_rows was overestimated
        for r_idx_unused in range(current_plot_row_idx, plot_rows):
            for c_idx_unused in range(num_cols):
                if r_idx_unused < axes.shape[0] and c_idx_unused < axes.shape[1]:
                    axes[r_idx_unused, c_idx_unused].axis('off')

        fig.tight_layout(rect=[0, 0, 1, 0.97])  # Adjusted rect for suptitle
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plot_filename = os.path.join(save_dir, f"feature_viz_sample_{i}_digit_{current_label}.png")
            fig.savefig(plot_filename);
            print(f"Feature viz saved: {plot_filename}")
        plt.show(block=False)


def plot_waveron_filters(model, save_dir=None):
    model.eval()
    if not (hasattr(model, 'waverons1') and model.waverons1 and hasattr(model.waverons1[0], 'conv')):
        print(
            f"Model {type(model).__name__} is not a Convolutional Waveron-based model or lacks 'waverons1' with 'conv'. Skipping filter plot.")
        return

    num_waverons1 = len(model.waverons1)
    conv_out_channels = model.waverons1[0].conv.out_channels
    kernel_size_tuple = model.waverons1[0].conv.kernel_size
    input_channels_to_waveron1 = model.waverons1[0].conv.in_channels

    if conv_out_channels == 0 or input_channels_to_waveron1 == 0:
        print("Waveron conv_out_channels or input_channels is 0. Skipping filter plot.")
        return

    num_filter_plots_per_waveron = conv_out_channels
    plot_in_channel_idx = 0
    if input_channels_to_waveron1 > 1:
        print(
            f"Warning: Plotting convolutional filters for input channel {plot_in_channel_idx} only (total input_channels: {input_channels_to_waveron1})")

    fig, axes = plt.subplots(num_waverons1, num_filter_plots_per_waveron,
                             figsize=(num_filter_plots_per_waveron * 2.2, num_waverons1 * 2.2), squeeze=False)
    fig.suptitle(
        f"W1 Convolutional Filters (Wavelet: {model.wavelet_name}, Kernel: {kernel_size_tuple[0]}x{kernel_size_tuple[1]}, InCh: {input_channels_to_waveron1}, OutCh: {conv_out_channels})",
        fontsize=10)
    initial_band_names_filt = ['LL0', 'LH0', 'HL0', 'HH0']

    for i in range(num_waverons1):
        filters = model.waverons1[i].conv.weight.data.detach().cpu()
        for j in range(num_filter_plots_per_waveron):
            if j < filters.shape[0] and plot_in_channel_idx < filters.shape[1]:
                filt = filters[j, plot_in_channel_idx, :, :];
                ax = axes[i, j]
                ax.imshow(filt, cmap='gray', vmin=filt.min(), vmax=filt.max())
                ax.set_xticks([]);
                ax.set_yticks([])
                if j == 0: ax.set_ylabel(f"W1 from\n{initial_band_names_filt[i]}", rotation=0, size='medium',
                                         labelpad=40)
                if i == num_waverons1 - 1: ax.set_xlabel(f"Filt {j}\n(InCh {plot_in_channel_idx})", fontsize=9)
            else:
                axes[i, j].axis('off')
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_filename = os.path.join(save_dir, "waveron_layer1_conv_filters.png")
        fig.savefig(plot_filename);
        print(f"W1 conv filters saved: {plot_filename}")
    plt.show(block=False)


def calculate_and_plot_average_subband_energies(model, dataloader, device, save_dir=None, dataset_name_for_plot="",
                                                model_type_for_plot=""):
    model.eval()
    if len(dataloader) == 0: print("Dataloader empty, skipping energy plot."); return
    if not (hasattr(model, 'forward') and 'return_intermediate_outputs' in model.forward.__code__.co_varnames):
        print(f"Model {type(model).__name__} no 'return_intermediate_outputs' for energy. Skip.");
        return
    is_single_layer = model.num_waveron_layers == 1 if hasattr(model, 'num_waveron_layers') else False
    num_initial_bands = 4;
    sum_initial_band_energies = [0.0] * num_initial_bands
    num_waveron1_paths = 4;
    num_waveron1_output_types = 4
    sum_waveron1_output_energies = [[0.0] * num_waveron1_output_types for _ in range(num_waveron1_paths)]
    total_samples_processed = 0
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs_tuple = model(images, return_intermediate_outputs=True)
            if not (isinstance(outputs_tuple, tuple) and len(outputs_tuple) == 2 and isinstance(outputs_tuple[1],
                                                                                                dict) and 'energy_intermediates' in
                    outputs_tuple[1]):
                print(f"Could not retrieve 'energy_intermediates' for {type(model).__name__}. Skipping batch.");
                continue
            energy_intermediates = outputs_tuple[1]['energy_intermediates']
            batch_size = images.size(0);
            total_samples_processed += batch_size
            if 'initial_bands' not in energy_intermediates or not energy_intermediates['initial_bands']:
                print("Required 'initial_bands' empty in energy_intermediates. Skipping batch.");
                continue
            initial_bands_for_energy = energy_intermediates['initial_bands']
            if len(initial_bands_for_energy) == num_initial_bands:
                for i_band, band_tensor in enumerate(initial_bands_for_energy):
                    sum_initial_band_energies[i_band] += torch.sum(
                        torch.mean(band_tensor.pow(2).view(batch_size, -1), dim=1)).item()
            if 'all_waveron1_path_outputs' in energy_intermediates and energy_intermediates[
                'all_waveron1_path_outputs']:
                all_w1_outputs_for_energy = energy_intermediates['all_waveron1_path_outputs']
                if len(all_w1_outputs_for_energy) == num_waveron1_paths:
                    for path_idx in range(num_waveron1_paths):
                        if isinstance(all_w1_outputs_for_energy[path_idx], list) and len(
                                all_w1_outputs_for_energy[path_idx]) == num_waveron1_output_types:
                            for band_type_idx in range(num_waveron1_output_types):
                                band_tensor = all_w1_outputs_for_energy[path_idx][band_type_idx]
                                sum_waveron1_output_energies[path_idx][band_type_idx] += torch.sum(
                                    torch.mean(band_tensor.pow(2).view(batch_size, -1), dim=1)).item()
    if total_samples_processed == 0: print("No samples processed for energy calc."); return
    avg_initial_band_energies = [s / total_samples_processed for s in sum_initial_band_energies]
    all_labels = [f"Initial {b}" for b in ['LL0', 'LH0', 'HL0', 'HH0']];
    all_energies = avg_initial_band_energies
    input_band_abbrevs = ['LL0', 'LH0', 'HL0', 'HH0'];
    output_band_abbrevs = ['LL', 'LH', 'HL', 'HH']
    if sum(sum(path_e) for path_e in sum_waveron1_output_energies) > 1e-9:
        avg_waveron1_output_energies = [[s / total_samples_processed for s in path_e] for path_e in
                                        sum_waveron1_output_energies]
        waveron1_output_labels = [f"W1({input_band_abbrevs[p]})->{output_band_abbrevs[b]}" for p in range(4) for b in
                                  range(4)]
        all_labels.extend(waveron1_output_labels);
        all_energies.extend([e for path_e in avg_waveron1_output_energies for e in path_e])
    plt.figure(figsize=(max(10, len(all_labels) * 0.6), 7))
    plt.bar(all_labels, all_energies,
            color=['skyblue'] * num_initial_bands + ['lightcoral'] * (len(all_labels) - num_initial_bands))
    plt.xticks(rotation=90, ha='right', fontsize=8);
    plt.ylabel("Average Energy (Mean Squared Value)")
    model_name_for_title = model.__class__.__name__
    wavelet_name_for_title = model.wavelet_name if hasattr(model, 'wavelet_name') else 'N/A'
    plt.title(
        f"Avg Sub-band Energies (Dataset: {dataset_name_for_plot}, Model: {model_name_for_title}, Wavelet: {wavelet_name_for_title})",
        fontsize=10)
    plt.grid(axis='y', linestyle='--');
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_filename = os.path.join(save_dir, "average_subband_energies.png")
        plt.savefig(plot_filename);
        print(f"Avg sub-band energies plot saved: {plot_filename}")
    plt.show(block=False)


def plot_fourier_network_analysis(model, dataloader, device, save_dir=None, dataset_name_for_plot=""):
    model.eval()
    if len(dataloader) == 0: print("Dataloader empty, skipping Fourier analysis plot."); return
    sum_radial_spectrum = None;
    radial_bins = None;
    num_radial_samples = 0
    if not hasattr(model, 'crop_h'):
        print(f"Model {type(model).__name__} is not FourierNet. Skipping Fourier analysis plot.")
        return
    crop_h = model.crop_h;
    crop_w = model.crop_w
    sum_cropped_spectrum_2d = torch.zeros((1, 1, crop_h, crop_w), device=device)
    num_samples_for_2d_avg = 0
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            current_batch_size = images.size(0)
            for i in range(current_batch_size):
                img = images[i];
                if img.shape[0] > 1: img = img[0:1, :, :]
                fft_coeffs = torch.fft.fft2(img)
                power_spectrum_2d = torch.abs(fft_coeffs) ** 2
                shifted_spectrum_2d = torch.fft.fftshift(power_spectrum_2d, dim=(-2, -1)).squeeze(0)
                h, w = shifted_spectrum_2d.shape
                center_h, center_w = h // 2, w // 2
                y_coords, x_coords = np.ogrid[:h, :w]
                r = np.sqrt((x_coords - center_w) ** 2 + (y_coords - center_h) ** 2);
                r = r.astype(int)
                max_r = int(np.max(r))
                if radial_bins is None:
                    radial_bins = np.arange(0, max_r + 2)
                    sum_radial_spectrum = np.zeros(len(radial_bins) - 1)
                hist, _ = np.histogram(r.ravel(), bins=radial_bins, weights=shifted_spectrum_2d.cpu().numpy().ravel())
                count_hist, _ = np.histogram(r.ravel(), bins=radial_bins)
                valid_bins = count_hist > 0
                current_image_radial_avg = np.zeros_like(sum_radial_spectrum)
                current_image_radial_avg[valid_bins] = hist[valid_bins] / count_hist[valid_bins]
                sum_radial_spectrum += current_image_radial_avg
                num_radial_samples += 1
            if hasattr(model, 'forward') and 'return_spectrum_features' in model.forward.__code__.co_varnames:
                outputs_tuple = model(images, return_spectrum_features=True)
                if isinstance(outputs_tuple, tuple) and len(outputs_tuple) == 2 and isinstance(outputs_tuple[1],
                                                                                               dict) and 'spectrum_features' in \
                        outputs_tuple[1]:
                    cropped_spectrum_batch, _ = outputs_tuple[1]['spectrum_features']
                    sum_cropped_spectrum_2d += torch.sum(cropped_spectrum_batch.to(device), dim=0, keepdim=True)
                    num_samples_for_2d_avg += current_batch_size
                else:
                    print(f"Could not retrieve 'spectrum_features' for Fourier plotting from {type(model).__name__}.")
            else:
                pass
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Fourier Analysis (Dataset: {dataset_name_for_plot})", fontsize=14)
    if num_radial_samples > 0:
        avg_radial_spectrum = sum_radial_spectrum / num_radial_samples
        axes[0].plot(radial_bins[:-1], avg_radial_spectrum)
        axes[0].set_title("Avg. 1D Radial Power Spectrum (Input)");
        axes[0].set_xlabel("Radial Frequency");
        axes[0].set_ylabel("Average Power")
        axes[0].set_yscale('log');
        axes[0].grid(True, which="both", ls="-", alpha=0.5)
    else:
        axes[0].text(0.5, 0.5, "No radial spectrum data", ha='center', va='center'); axes[0].set_title(
            "Avg. 1D Radial Power Spectrum (Input)")
    if num_samples_for_2d_avg > 0 and hasattr(model, 'crop_h'):
        avg_cropped_spectrum_2d = sum_cropped_spectrum_2d.squeeze().cpu().numpy() / num_samples_for_2d_avg
        im = axes[1].imshow(np.log1p(avg_cropped_spectrum_2d), cmap='viridis', aspect='auto',
                            extent=[0, avg_cropped_spectrum_2d.shape[1], avg_cropped_spectrum_2d.shape[0], 0])
        axes[1].set_title(
            f"Avg. Log Cropped 2D Spectrum\n(FourierNet Feat - {avg_cropped_spectrum_2d.shape[0]}x{avg_cropped_spectrum_2d.shape[1]})")
        axes[1].set_xlabel("Frequency Comp (W)");
        axes[1].set_ylabel("Frequency Comp (H)")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        axes[1].text(0.5, 0.5, "No cropped spectrum data\n(or not FourierNet)", ha='center', va='center'); axes[
            1].set_title(f"Avg. Log Cropped 2D Spectrum (FourierNet Feature)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_filename = os.path.join(save_dir, "fourier_network_analysis.png")
        plt.savefig(plot_filename);
        print(f"Fourier network analysis plot saved to {plot_filename}")
    plt.show(block=False)


def plot_feature_space_projection(features, labels, model_type, save_dir=None, dataset_name_for_plot="", n_components=2,
                                  perplexity=30.0, n_iter=1000):
    if features.shape[0] == 0: print(f"No features provided for t-SNE plot of {model_type}. Skipping."); return
    actual_perplexity = perplexity
    if features.shape[0] <= perplexity:
        print(f"Warning: Samples ({features.shape[0]}) <= perplexity ({perplexity}). Adjusting perplexity.")
        actual_perplexity = max(5.0, float(features.shape[0] - 1))
        if features.shape[0] <= 5: print(
            f"Not enough samples ({features.shape[0]}) for a meaningful t-SNE plot. Skipping."); return
    print(f"Running t-SNE for {model_type} on {features.shape[0]} samples with perplexity {actual_perplexity}...")
    tsne = TSNE(n_components=n_components, perplexity=actual_perplexity, n_iter=n_iter, init='pca',
                learning_rate='auto', random_state=42)
    features_2d = tsne.fit_transform(features)
    plt.figure(figsize=(10, 8))
    cmap = plt.cm.get_cmap('viridis', np.unique(labels).size) if np.unique(labels).size <= 20 else 'viridis'
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap=cmap, alpha=0.7, s=10)
    try:
        unique_labels = np.unique(labels)
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=str(l),
                              markerfacecolor=cmap(idx / len(unique_labels) if len(unique_labels) > 1 else 0.5),
                              markersize=5) for idx, l in enumerate(unique_labels)]
        plt.legend(handles=handles, title="Classes")
    except Exception as e:
        print(f"Error creating t-SNE legend: {e}. Using default legend."); plt.legend(title="Classes")
    plt.title(f"t-SNE Projection of Penultimate Features\n({model_type} on {dataset_name_for_plot})")
    plt.xlabel("t-SNE Component 1");
    plt.ylabel("t-SNE Component 2");
    plt.grid(True, linestyle='--', alpha=0.5)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_filename = os.path.join(save_dir, f"tsne_features_{model_type}.png")
        plt.savefig(plot_filename);
        print(f"t-SNE plot for {model_type} saved to {plot_filename}")
    plt.show(block=False)


def plot_accuracy_vs_noise(noise_levels_list, all_models_accuracies_dict, save_dir, training_condition_on_plot):
    plt.figure(figsize=(10, 7))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    marker_idx = 0
    for model_tag, accuracies in all_models_accuracies_dict.items():
        if len(noise_levels_list) == len(accuracies):
            plt.plot(noise_levels_list, accuracies, marker=markers[marker_idx % len(markers)], linestyle='-',
                     label=model_tag)
            marker_idx += 1
        else:
            print(f"Warning: Mismatch in length of noise_levels and accuracies for {model_tag} in noise plot.")
    plt.title(f'Model Robustness: Accuracy vs. Noise Level\n({training_condition_on_plot}, Evaluated on Noisy MNIST)')
    plt.xlabel('Noise Standard Deviation on Test Set');
    plt.ylabel('Test Accuracy (%)')
    if noise_levels_list: plt.xticks(noise_levels_list)
    plt.ylim(0, 101);
    plt.legend(loc='best', fontsize='small');
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_filename = os.path.join(save_dir, f"COMBINED_accuracy_vs_noise.png")
        plt.savefig(plot_filename);
        print(f"Combined Accuracy vs. Noise plot saved: {plot_filename}")
    plt.show(block=False)


def plot_accuracy_vs_bg_threshold(bg_thresholds_list, all_models_accuracies_dict, save_dir, training_condition_on_plot):
    plt.figure(figsize=(10, 7))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    marker_idx = 0
    for model_tag, accuracies in all_models_accuracies_dict.items():
        if len(bg_thresholds_list) == len(accuracies):
            plt.plot(bg_thresholds_list, accuracies, marker=markers[marker_idx % len(markers)], linestyle='-',
                     label=model_tag)
            marker_idx += 1
        else:
            print(f"Warning: Mismatch for {model_tag} in bg_threshold plot.")
    plt.title(
        f'Model Robustness: Accuracy vs. Background Randomization Threshold\n({training_condition_on_plot}, Evaluated on MNIST with Random BG)')
    plt.xlabel('Background Randomization Threshold');
    plt.ylabel('Test Accuracy (%)')
    if bg_thresholds_list: plt.xticks(bg_thresholds_list)
    plt.ylim(0, 101);
    plt.legend(loc='best', fontsize='small');
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_filename = os.path.join(save_dir, f"COMBINED_accuracy_vs_bg_threshold.png")
        plt.savefig(plot_filename);
        print(f"Combined Accuracy vs. BG Threshold plot saved: {plot_filename}")
    plt.show(block=False)


# --- NEW: Plot for Waveron Final Subband Spatial Energy Maps ---
def plot_waveron_final_subband_maps_avg_energy(model, dataloader, device, save_dir=None, dataset_name_for_plot=""):
    model.eval()
    if len(dataloader) == 0: print("Dataloader empty, skipping final subband energy plot."); return
    if not (hasattr(model, 'forward') and 'return_final_subband_maps' in model.forward.__code__.co_varnames):
        print(f"Model {type(model).__name__} does not support 'return_final_subband_maps'. Skipping plot.")
        return

    # Determine the number of final subbands (should be 16 for current Waveron architectures)
    # and their spatial dimensions by doing a dry run with one batch.
    temp_images, _ = next(iter(dataloader))
    temp_images = temp_images.to(device)
    with torch.no_grad():
        _, temp_optional_returns = model(temp_images, return_final_subband_maps=True)

    if not (temp_optional_returns and 'final_subband_maps' in temp_optional_returns and temp_optional_returns[
        'final_subband_maps']):
        print("Could not retrieve 'final_subband_maps' from model for dimension check. Skipping plot.")
        return

    sample_final_bands = temp_optional_returns['final_subband_maps']
    num_final_bands = len(sample_final_bands)
    if num_final_bands == 0:
        print("No final subband maps returned by the model. Skipping plot.")
        return

    # Assuming all final subbands have the same H, W, C after their respective Waveron processing
    # For this plot, we average over channels within each of the 16 bands if C > 1
    # The "energy" here will be the mean of the (potentially multi-channel) feature map.
    # Or, more like an "average activation map".

    # Initialize accumulators for the sum of feature maps (to later average)
    # Each element in this list will be a 2D tensor (H_final, W_final)
    sum_final_subband_maps = [torch.zeros_like(band[0, 0, :, :].cpu()) for band in
                              sample_final_bands]  # Use first sample's 0th channel shape

    total_samples_processed = 0

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            _, optional_returns = model(images, return_final_subband_maps=True)

            if not (optional_returns and 'final_subband_maps' in optional_returns and optional_returns[
                'final_subband_maps']):
                print("Warning: 'final_subband_maps' not found in batch. Skipping batch for this plot.")
                continue

            final_subband_maps_batch = optional_returns['final_subband_maps']  # List of 16 tensors [B, C, H, W]

            current_batch_size = images.size(0)
            total_samples_processed += current_batch_size

            for band_idx in range(num_final_bands):
                # For each of the 16 bands, average over batch and channels to get a [H, W] map
                # Then sum these [H, W] maps across batches.
                # band_tensor is [B, C, H, W]
                band_tensor = final_subband_maps_batch[band_idx].cpu()  # Move to CPU
                # Take mean over channel dimension first, then sum over batch dimension
                # This gives sum of [H,W] maps.
                sum_final_subband_maps[band_idx] += torch.sum(torch.mean(band_tensor, dim=1), dim=0)

    if total_samples_processed == 0:
        print("No samples processed for final subband maps. Skipping plot.")
        return

    avg_final_subband_maps = [(s_map / total_samples_processed) for s_map in sum_final_subband_maps]

    # Plotting in a 4x4 grid
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(
        f"Average Spatial Energy/Activation in Final Waveron Sub-bands (Input to FC)\n(Model: {model.__class__.__name__}, Wavelet: {model.wavelet_name if hasattr(model, 'wavelet_name') else 'N/A'}, Data: {dataset_name_for_plot})",
        fontsize=12, y=0.97)

    band_counter = 0
    # Define more descriptive labels if possible, based on the architecture
    # For now, just use generic band indices
    for r in range(4):
        for c in range(4):
            if band_counter < len(avg_final_subband_maps):
                ax = axes[r, c]
                im = ax.imshow(avg_final_subband_maps[band_counter].numpy(), cmap='viridis')
                ax.set_title(f"Final Band {band_counter + 1}", fontsize=9)
                ax.axis('off')
                # plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04) # Optional colorbar
            else:
                axes[r, c].axis('off')  # Turn off unused subplots
            band_counter += 1

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_filename = os.path.join(save_dir, "waveron_final_subband_avg_energy_maps.png")
        plt.savefig(plot_filename)
        print(f"Waveron final subband average energy maps saved to {plot_filename}")
    plt.show(block=False)

