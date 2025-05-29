import torch
import torch.optim as optim
import torch.nn as nn
import os
from datetime import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt

from networks.waveron_network import WaveNetClassifier, WaveNetClassifierConcat, SingleLayerWaveronConcat
from networks.mlp_network import MLP
from networks.fourier_network import FourierNet

from utils.dataset_loader import get_dataset
from utils.plotting import (
    plot_training_metrics,
    plot_waveron_features,
    plot_waveron_filters,
    calculate_and_plot_average_subband_energies,
    plot_fourier_network_analysis,
    plot_feature_space_projection,
    plot_accuracy_vs_noise,
    plot_accuracy_vs_bg_threshold
)


def get_penultimate_features(model, dataloader, device, model_type_str, num_samples=1000):
    model.eval()
    all_features_list = []
    all_labels_list = []
    samples_collected = 0
    if len(dataloader) == 0: return np.array([]), np.array([])

    with torch.no_grad():
        for images, labels in dataloader:
            if samples_collected >= num_samples: break
            images_batch = images.to(device)
            current_batch_size = images_batch.size(0)
            remaining_samples = num_samples - samples_collected
            if current_batch_size > remaining_samples:
                images_batch = images_batch[:remaining_samples]
                labels_batch_subset = labels[:remaining_samples]
                current_batch_size = remaining_samples
            else:
                labels_batch_subset = labels

            penultimate_features_tensor = None
            outputs_tuple = None

            if model_type_str == 'mlp':
                images_for_model = images_batch.view(current_batch_size, -1)
                outputs_tuple = model(images_for_model, return_penultimate_features=True)
                if isinstance(outputs_tuple, tuple) and len(outputs_tuple) == 2:
                    penultimate_features_tensor = outputs_tuple[1]
            elif model_type_str == 'fourier':
                outputs_tuple = model(images_batch, return_penultimate_features=True)
                if isinstance(outputs_tuple, tuple) and len(outputs_tuple) == 2 and isinstance(outputs_tuple[1], dict):
                    penultimate_features_tensor = outputs_tuple[1].get('penultimate_features')
            elif model_type_str in ['waveron', 'waveron_concat', 'single_layer_waveron_concat']:
                outputs_tuple = model(images_batch, return_penultimate_features=True)
                if isinstance(outputs_tuple, tuple) and len(outputs_tuple) == 2 and isinstance(outputs_tuple[1], dict):
                    penultimate_features_tensor = outputs_tuple[1].get('penultimate_features')
            else:
                raise ValueError(f"Unknown model type for feature extraction: {model_type_str}")

            if penultimate_features_tensor is not None:
                all_features_list.append(penultimate_features_tensor.cpu().numpy())
                all_labels_list.append(labels_batch_subset.cpu().numpy())
                samples_collected += current_batch_size
            elif current_batch_size > 0:
                print(f"Penultimate features are None for a batch from {model_type_str}. Model output: {outputs_tuple}")

    if not all_features_list: return np.array([]), np.array([])
    all_features = np.concatenate(all_features_list, axis=0)
    all_labels = np.concatenate(all_labels_list, axis=0)
    return all_features, all_labels


def main(args):
    print(f"Starting experiment with arguments: {args}")

    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # --- Determine Top-Level Experiment Directory Name ---
    experiment_tag = args.dataset
    if args.eval_wavelets_robustness:
        experiment_tag = f"wavelet_robustness_on_mnist_noisy_eval"
        if args.model_types and args.model_types[0] in ['waveron', 'waveron_concat', 'single_layer_waveron_concat']:
            experiment_tag = f"{args.model_types[0]}_wavelet_robustness_eval"  # Use the specific waveron model type
    elif args.eval_noise_levels:
        experiment_tag = f"mnist_clean_train_eval_multi_noise"
    elif args.eval_bg_rand_thresholds:
        experiment_tag = f"mnist_clean_train_eval_multi_bgrand"
    elif args.dataset == 'mnist_noisy':
        experiment_tag = f"mnist_train_noisy_std{args.noise_std}"
    elif args.dataset == 'mnist_bg_rand':
        experiment_tag = f"mnist_train_bg_rand_thr{args.bg_rand_threshold}"
    elif args.dataset == 'mnist_noisy_bg_rand':
        experiment_tag = f"mnist_train_noisy_std{args.noise_std}_bg_rand_thr{args.bg_rand_threshold}"

    experiment_base_dir = os.path.join("results", f"{experiment_tag}_{current_time_str}")
    os.makedirs(experiment_base_dir, exist_ok=True)
    print(f"All results for this run will be saved under: {experiment_base_dir}")

    all_models_noise_evaluation_accuracies = {}
    all_models_bg_rand_evaluation_accuracies = {}

    # --- Determine Wavelet Iteration List ---
    wavelet_iteration_list = [args.wavelet]  # Default to single wavelet from args
    if args.eval_wavelets_robustness:
        wavelet_iteration_list = ['haar', 'db4', 'sym4', 'bior4.4', 'coif1', 'rbio2.2']
        print(f"Wavelet Robustness Evaluation Mode: Testing wavelets: {wavelet_iteration_list}")
        if not (args.model_types and args.model_types[0] in ['waveron', 'waveron_concat',
                                                             'single_layer_waveron_concat']):
            print(
                "Error: --eval_wavelets_robustness is selected, but the first model in --model_types is not a Waveron variant. Please specify a Waveron model type first.")
            return

    for current_model_type_arg in args.model_types:

        # If evaluating wavelets, only process the first (assumed Waveron) model type with different wavelets
        if args.eval_wavelets_robustness and current_model_type_arg not in ['waveron', 'waveron_concat',
                                                                            'single_layer_waveron_concat']:
            print(f"Skipping model type {current_model_type_arg} during wavelet robustness evaluation.")
            continue

        # Inner loop for wavelets if eval_wavelets_robustness is on
        # Otherwise, this loop runs once with args.wavelet
        for current_wavelet_for_model in (
        wavelet_iteration_list if current_model_type_arg in ['waveron', 'waveron_concat',
                                                             'single_layer_waveron_concat'] else [args.wavelet]):

            print(
                f"\n\n{'=' * 20} Processing Model: {current_model_type_arg.upper()}, Wavelet: {current_wavelet_for_model or 'N/A'} {'=' * 20}")

            # --- Training Dataset ---
            training_dataset_name = args.dataset
            if args.eval_noise_levels or args.eval_bg_rand_thresholds or args.eval_wavelets_robustness:
                print("Special evaluation mode: Training will be on clean MNIST.")
                training_dataset_name = 'mnist'

            actual_training_noise_std = 0.0 if (
                        args.eval_noise_levels or args.eval_bg_rand_thresholds or args.eval_wavelets_robustness) else (
                args.noise_std if 'noisy' in training_dataset_name else 0.0)
            actual_training_bg_thresh = 0.0 if (
                        args.eval_noise_levels or args.eval_bg_rand_thresholds or args.eval_wavelets_robustness) else (
                args.bg_rand_threshold if 'bg_rand' in training_dataset_name else 0.0)

            print(
                f"Loading training dataset: {training_dataset_name} (Noise: {actual_training_noise_std}, BG: {actual_training_bg_thresh}) ...")
            train_loader, _, num_classes, input_size_hw, input_channels = get_dataset(
                dataset_name=training_dataset_name, data_root=args.data_root,
                target_img_size=(args.target_img_size, args.target_img_size), batch_size=args.batch_size,
                noise_std=actual_training_noise_std, bg_rand_threshold=actual_training_bg_thresh
            )
            if train_loader is None: print(f"Failed to load training dataset. Skipping."); continue

            _, clean_test_loader_for_plots, _, _, _ = get_dataset('mnist', batch_size=args.batch_size,
                                                                  data_root=args.data_root, target_img_size=(28, 28))
            if clean_test_loader_for_plots is None: print("Warning: Failed to load clean MNIST test set for plots.")

            # --- Model-Specific Tags and Save Directory ---
            model_specific_tag_for_dir = f"{current_model_type_arg}"
            model_specific_tag_for_legend = f"{current_model_type_arg}"

            if current_model_type_arg in ['waveron', 'waveron_concat', 'single_layer_waveron_concat']:
                current_w = current_wavelet_for_model  # Use the wavelet from the loop
                model_specific_tag_for_dir += f"_{current_w}_k{args.kernel_size}_ch{args.waveron_channels}"
                model_specific_tag_for_legend = model_specific_tag_for_dir
            elif current_model_type_arg == 'mlp':
                model_specific_tag_for_dir += f"_h{'_'.join(map(str, args.mlp_hidden_dims))}"
                model_specific_tag_for_legend = model_specific_tag_for_dir
            elif current_model_type_arg == 'fourier':
                model_specific_tag_for_dir += f"_crop{args.fourier_crop_fraction}_h{'_'.join(map(str, args.fourier_mlp_hidden_dims))}"
                model_specific_tag_for_legend = model_specific_tag_for_dir

            model_save_directory = os.path.join(experiment_base_dir, model_specific_tag_for_dir)
            os.makedirs(model_save_directory, exist_ok=True)
            print(f"Plots for {model_specific_tag_for_dir} will be saved in: {model_save_directory}")

            device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
            model = None
            if current_model_type_arg == 'waveron':
                model = WaveNetClassifier(num_classes=num_classes, wavelet_name=current_wavelet_for_model,
                                          kernel_size=args.kernel_size,
                                          waveron_conv_channels=args.waveron_channels, input_size_hw=input_size_hw,
                                          input_channels=input_channels).to(device)
            elif current_model_type_arg == 'waveron_concat':
                model = WaveNetClassifierConcat(num_classes=num_classes, wavelet_name=current_wavelet_for_model,
                                                kernel_size=args.kernel_size,
                                                waveron_conv_channels=args.waveron_channels,
                                                input_size_hw=input_size_hw,
                                                input_channels=input_channels).to(device)
            elif current_model_type_arg == 'single_layer_waveron_concat':
                model = SingleLayerWaveronConcat(num_classes=num_classes, wavelet_name=current_wavelet_for_model,
                                                 kernel_size=args.kernel_size,
                                                 waveron_conv_channels=args.waveron_channels,
                                                 input_size_hw=input_size_hw,
                                                 input_channels=input_channels).to(device)
            elif current_model_type_arg == 'mlp':
                flattened_input_dim = input_size_hw[0] * input_size_hw[1] * input_channels
                model = MLP(input_dim=flattened_input_dim, hidden_dims=args.mlp_hidden_dims, output_dim=num_classes).to(
                    device)
            elif current_model_type_arg == 'fourier':
                model = FourierNet(input_size_hw=input_size_hw, input_channels=input_channels,
                                   crop_fraction=args.fourier_crop_fraction,
                                   mlp_hidden_dims=args.fourier_mlp_hidden_dims, output_dim=num_classes).to(device)

            print(model)
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total trainable parameters for {model_specific_tag_for_legend}: {total_params:,}")

            criterion = nn.CrossEntropyLoss();
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            train_losses_hist, test_losses_hist, test_accuracies_hist = [], [], []

            print(f"\n--- Training {model_specific_tag_for_legend} on {training_dataset_name} ---")
            for epoch in range(args.epochs):
                model.train();
                epoch_train_loss = 0.0
                for batch_idx, (images, labels) in enumerate(train_loader):
                    images, labels = images.to(device), labels.to(device)
                    if current_model_type_arg == 'mlp': images = images.view(images.size(0), -1)
                    outputs = model(images);
                    if isinstance(outputs, tuple): outputs = outputs[0]
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad();
                    loss.backward();
                    optimizer.step()
                    epoch_train_loss += loss.item()
                    if (batch_idx + 1) % 50 == 0 or batch_idx == len(train_loader) - 1: print(
                        f'M [{model_specific_tag_for_legend}] E [{epoch + 1}/{args.epochs}], S [{batch_idx + 1}/{len(train_loader)}], L: {loss.item():.4f}')
                avg_epoch_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else 0
                train_losses_hist.append(avg_epoch_train_loss);
                print(
                    f'M [{model_specific_tag_for_legend}] E [{epoch + 1}/{args.epochs}] Avg Train L: {avg_epoch_train_loss:.4f}')
                model.eval();
                epoch_test_loss, correct_preds, total_samples_eval = 0.0, 0, 0
                if clean_test_loader_for_plots:
                    with torch.no_grad():
                        for images, labels in clean_test_loader_for_plots:
                            images, labels = images.to(device), labels.to(device)
                            if current_model_type_arg == 'mlp': images = images.view(images.size(0), -1)
                            outputs = model(images);
                            if isinstance(outputs, tuple): outputs = outputs[0]
                            loss = criterion(outputs, labels);
                            epoch_test_loss += loss.item()
                            _, predicted = torch.max(outputs.data, 1)
                            total_samples_eval += labels.size(0);
                            correct_preds += (predicted == labels).sum().item()
                    avg_epoch_test_loss = epoch_test_loss / len(clean_test_loader_for_plots) if len(
                        clean_test_loader_for_plots) > 0 else 0
                    accuracy = 100 * correct_preds / total_samples_eval if total_samples_eval > 0 else 0
                    test_losses_hist.append(avg_epoch_test_loss);
                    test_accuracies_hist.append(accuracy)
                    print(
                        f'M [{model_specific_tag_for_legend}] E [{epoch + 1}/{args.epochs}] Clean Test L: {avg_epoch_test_loss:.4f}, Clean Test Acc: {accuracy:.2f}%')
            print(f'--- Finished Training for {model_specific_tag_for_legend} ---')

            plot_training_metrics(train_losses_hist, test_losses_hist, test_accuracies_hist, args.epochs,
                                  save_dir=model_save_directory)
            if clean_test_loader_for_plots and len(clean_test_loader_for_plots) > 0:
                features, true_labels = get_penultimate_features(model, clean_test_loader_for_plots, device,
                                                                 current_model_type_arg, num_samples=args.tsne_samples)
                if features.shape[0] > 0: plot_feature_space_projection(features, true_labels,
                                                                        model_specific_tag_for_legend,
                                                                        save_dir=model_save_directory,
                                                                        dataset_name_for_plot=f"Clean Test (Trained on {training_dataset_name})")

            if current_model_type_arg in ['waveron', 'waveron_concat', 'single_layer_waveron_concat']:
                if total_params < 2_000_000: plot_waveron_filters(model, save_dir=model_save_directory)
                if clean_test_loader_for_plots and len(clean_test_loader_for_plots) > 0:
                    plot_waveron_features(model, clean_test_loader_for_plots, device,
                                          num_samples=min(2, args.batch_size), save_dir=model_save_directory,
                                          dataset_name_for_plot=f"Clean Test (Trained on {training_dataset_name})")
                    calculate_and_plot_average_subband_energies(model, clean_test_loader_for_plots, device,
                                                                save_dir=model_save_directory,
                                                                dataset_name_for_plot=f"Clean Test (Trained on {training_dataset_name})",
                                                                model_type_for_plot=current_model_type_arg)
            elif current_model_type_arg == 'fourier':
                if clean_test_loader_for_plots and len(clean_test_loader_for_plots) > 0:
                    plot_fourier_network_analysis(model, clean_test_loader_for_plots, device,
                                                  save_dir=model_save_directory,
                                                  dataset_name_for_plot=f"Clean Test (Trained on {training_dataset_name})")

            if args.eval_noise_levels:
                current_model_noise_accuracies = []
                for noise_level_eval in args.eval_noise_levels:
                    _, noisy_test_loader, _, _, _ = get_dataset('mnist_noisy', batch_size=args.batch_size,
                                                                noise_std=noise_level_eval, data_root=args.data_root,
                                                                target_img_size=(28, 28))
                    if noisy_test_loader is None: current_model_noise_accuracies.append(0); continue
                    model.eval();
                    correct_preds_noisy, total_samples_noisy = 0, 0
                    with torch.no_grad():
                        for images, labels in noisy_test_loader:
                            images, labels = images.to(device), labels.to(device)
                            if current_model_type_arg == 'mlp': images = images.view(images.size(0), -1)
                            outputs = model(images)
                            if isinstance(outputs, tuple): outputs = outputs[0]
                            _, predicted = torch.max(outputs.data, 1)
                            total_samples_noisy += labels.size(0);
                            correct_preds_noisy += (predicted == labels).sum().item()
                    accuracy_noisy = 100 * correct_preds_noisy / total_samples_noisy if total_samples_noisy > 0 else 0
                    current_model_noise_accuracies.append(accuracy_noisy)
                    print(
                        f"Acc for {model_specific_tag_for_legend} on noisy MNIST (std={noise_level_eval}): {accuracy_noisy:.2f}%")
                all_models_noise_evaluation_accuracies[model_specific_tag_for_legend] = current_model_noise_accuracies

            if args.eval_bg_rand_thresholds:  # This part remains, but might not be primary focus if eval_wavelets is on
                current_model_bg_rand_accuracies = []
                for bg_thresh_eval in args.eval_bg_rand_thresholds:
                    _, bg_rand_test_loader, _, _, _ = get_dataset('mnist_bg_rand', batch_size=args.batch_size,
                                                                  bg_rand_threshold=bg_thresh_eval,
                                                                  data_root=args.data_root, target_img_size=(28, 28))
                    if bg_rand_test_loader is None: current_model_bg_rand_accuracies.append(0); continue
                    model.eval();
                    correct_preds_bg, total_samples_bg = 0, 0
                    with torch.no_grad():
                        for images, labels in bg_rand_test_loader:
                            images, labels = images.to(device), labels.to(device)
                            if current_model_type_arg == 'mlp': images = images.view(images.size(0), -1)
                            outputs = model(images)
                            if isinstance(outputs, tuple): outputs = outputs[0]
                            _, predicted = torch.max(outputs.data, 1)
                            total_samples_bg += labels.size(0);
                            correct_preds_bg += (predicted == labels).sum().item()
                    accuracy_bg_rand = 100 * correct_preds_bg / total_samples_bg if total_samples_bg > 0 else 0
                    current_model_bg_rand_accuracies.append(accuracy_bg_rand)
                    print(
                        f"Acc for {model_specific_tag_for_legend} on MNIST w/ BG_Rand (thr={bg_thresh_eval}): {accuracy_bg_rand:.2f}%")
                all_models_bg_rand_evaluation_accuracies[
                    model_specific_tag_for_legend] = current_model_bg_rand_accuracies

            # If not evaluating wavelets robustness, break from inner wavelet loop (it only had one item: args.wavelet)
            if not (current_model_type_arg in ['waveron', 'waveron_concat',
                                               'single_layer_waveron_concat'] and args.eval_wavelets_robustness):
                break
                # End of outer loop (model_types) / inner loop (wavelets)

    if args.eval_noise_levels and all_models_noise_evaluation_accuracies:
        print("\n--- Generating Combined Accuracy vs. Noise Plot for all model/wavelet variants ---")
        plot_accuracy_vs_noise(noise_levels_list=args.eval_noise_levels,
                               all_models_accuracies_dict=all_models_noise_evaluation_accuracies,
                               save_dir=experiment_base_dir,
                               dataset_name_for_plot=args.dataset)

    plt.show()
    print(f"--- Experiment Finished. All results saved under {experiment_base_dir} ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Waveron Network Research Experiment")

    parser.add_argument('--model_types', type=str, nargs='+', default=['waveron'],
                        choices=['waveron', 'waveron_concat', 'single_layer_waveron_concat', 'mlp', 'fourier'],
                        help='List of model types to train and evaluate. If --eval_wavelets_robustness is set, this should primarily contain Waveron-type models.')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='Dataset to train on (e.g., mnist, mnist_noisy, mnist_bg_rand). This determines training conditions if not overridden by an eval_ flag.')
    parser.add_argument('--data_root', type=str, default='./data', help='Root directory for datasets')
    parser.add_argument('--target_img_size', type=int, default=28, help='Target image H/W (MNIST uses 28)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--tsne_samples', type=int, default=1000, help='Number of samples for t-SNE plot')

    parser.add_argument('--noise_std', type=float, default=0.2,
                        help='Std dev for Gaussian noise if training on mnist_noisy or mnist_noisy_bg_rand.')
    parser.add_argument('--bg_rand_threshold', type=float, default=0.1,
                        help='Threshold for AddRandomBackground if training on mnist_bg_rand or mnist_noisy_bg_rand.')

    parser.add_argument('--eval_noise_levels', type=float, nargs='*', default=[0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        help='List of noise std deviations for robustness evaluation. Trains on clean MNIST by default if this is set.')
    parser.add_argument('--eval_bg_rand_thresholds', type=float, nargs='*', default=None,
                        help='List of bg_rand thresholds for robustness evaluation. Trains on clean MNIST by default if this is set.')
    parser.add_argument('--eval_wavelets_robustness', action='store_true', default=True,
                        help='If set, runs robustness evaluation for multiple wavelets (defined in script) on specified Waveron models, training them on clean MNIST and evaluating on noisy MNIST defined by --eval_noise_levels.')

    parser.add_argument('--wavelet', type=str, default='db4',
                        help='Default wavelet type for Waveron network (used if not in --eval_wavelets_robustness mode).')
    parser.add_argument('--kernel_size', type=int, default=3, help='Conv kernel size in Waverons')
    parser.add_argument('--waveron_channels', type=int, default=8, help='Output channels for Waveron convolutions')

    parser.add_argument('--mlp_hidden_dims', type=int, nargs='+', default=[512, 256],
                        help='Hidden layer dimensions for standalone MLP')

    parser.add_argument('--fourier_crop_fraction', type=float, default=0.5,
                        help='Fraction of spectrum to crop for FourierNet')
    parser.add_argument('--fourier_mlp_hidden_dims', type=int, nargs='+', default=[256, 128],
                        help='Hidden layer dimensions for FourierNet MLP backend')

    args = main_args = parser.parse_args()

    main(main_args)

for dir_name in ["networks", "utils", "results"]:
    if not os.path.exists(dir_name): os.makedirs(dir_name)
    init_file = os.path.join(dir_name, "__init__.py")
    if not os.path.exists(init_file): open(init_file, "w").close()