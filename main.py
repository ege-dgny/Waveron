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
    dataset_folder_name_tag = args.dataset
    if args.eval_noise_levels:
        dataset_folder_name_tag = f"mnist_clean_train_eval_noisy"
    elif args.eval_bg_rand_thresholds:
        dataset_folder_name_tag = f"mnist_clean_train_eval_bg_rand"
    elif args.dataset == 'mnist_noisy':
        dataset_folder_name_tag = f"mnist_noisy_train_std{args.noise_std}"
    elif args.dataset == 'mnist_bg_rand':
        dataset_folder_name_tag = f"mnist_bg_rand_train_thr{args.bg_rand_threshold}"
    elif args.dataset == 'mnist_noisy_bg_rand':
        dataset_folder_name_tag = f"mnist_noisy_train_std{args.noise_std}_bg_rand_train_thr{args.bg_rand_threshold}"

    experiment_base_dir = os.path.join("results", f"{dataset_folder_name_tag}_{current_time_str}")
    os.makedirs(experiment_base_dir, exist_ok=True)
    print(f"All results for this run will be saved under: {experiment_base_dir}")

    all_models_noise_evaluation_accuracies = {}
    all_models_bg_rand_evaluation_accuracies = {}

    for current_model_type_arg in args.model_types:
        print(f"\n\n{'=' * 20} Processing Model Type: {current_model_type_arg.upper()} {'=' * 20}")

        training_dataset_name = args.dataset

        if current_model_type_arg != ("waveron" or "waveron_concat"):
            training_dataset_name = 'mnist'

        print(f"Loading training dataset: {training_dataset_name} for model {current_model_type_arg}...")
        current_train_noise_std = args.noise_std if 'noisy' in training_dataset_name else 0.0
        current_train_bg_thresh = args.bg_rand_threshold if 'bg_rand' in training_dataset_name else 0.0

        train_loader, clean_test_loader, num_classes, input_size_hw, input_channels = get_dataset(
            dataset_name=training_dataset_name, data_root=args.data_root,
            target_img_size=(args.target_img_size, args.target_img_size), batch_size=args.batch_size,
            noise_std=current_train_noise_std, bg_rand_threshold=current_train_bg_thresh
        )
        if train_loader is None:
            print(
                f"Failed to load training dataset {training_dataset_name} for model {current_model_type_arg}. Skipping this model.")
            continue

        if not clean_test_loader and (
                args.eval_noise_levels or args.eval_bg_rand_thresholds or training_dataset_name != 'mnist'):
            _, clean_test_loader, _, _, _ = get_dataset('mnist', data_root=args.data_root, target_img_size=(28, 28),
                                                        batch_size=args.batch_size)

        model_specific_tag_for_dir = f"{current_model_type_arg}"
        model_specific_tag_for_legend = f"{current_model_type_arg}"

        if current_model_type_arg in ['waveron', 'waveron_concat', 'single_layer_waveron_concat']:
            model_specific_tag_for_dir += f"_{args.wavelet}_k{args.kernel_size}_ch{args.waveron_channels}"  # Re-added kernel_size
            model_specific_tag_for_legend = model_specific_tag_for_dir
        elif current_model_type_arg == 'mlp':
            model_specific_tag_for_dir += f"_h{'_'.join(map(str, args.mlp_hidden_dims))}"
            model_specific_tag_for_legend = model_specific_tag_for_dir
        elif current_model_type_arg == 'fourier':
            model_specific_tag_for_dir += f"_crop{args.fourier_crop_fraction}_h{'_'.join(map(str, args.fourier_mlp_hidden_dims))}"
            model_specific_tag_for_legend = model_specific_tag_for_dir

        model_save_directory = os.path.join(experiment_base_dir, model_specific_tag_for_dir)
        os.makedirs(model_save_directory, exist_ok=True)
        print(f"Plots for model {current_model_type_arg} will be saved in: {model_save_directory}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
        print(f"Using device: {device}")

        print(f"Initializing model: {current_model_type_arg}...")
        model = None
        if current_model_type_arg == 'waveron':
            model = WaveNetClassifier(num_classes=num_classes, wavelet_name=args.wavelet,
                                      kernel_size=args.kernel_size,  # Added kernel_size
                                      waveron_conv_channels=args.waveron_channels,
                                      # This is conv_out_channels for Waveron
                                      input_size_hw=input_size_hw,
                                      input_channels=input_channels).to(device)
        elif current_model_type_arg == 'waveron_concat':
            model = WaveNetClassifierConcat(num_classes=num_classes, wavelet_name=args.wavelet,
                                            kernel_size=args.kernel_size,  # Added kernel_size
                                            waveron_conv_channels=args.waveron_channels,
                                            input_size_hw=input_size_hw,
                                            input_channels=input_channels).to(device)
        elif current_model_type_arg == 'single_layer_waveron_concat':
            model = SingleLayerWaveronConcat(num_classes=num_classes, wavelet_name=args.wavelet,
                                             kernel_size=args.kernel_size,  # Added kernel_size
                                             waveron_conv_channels=args.waveron_channels,
                                             input_size_hw=input_size_hw,
                                             input_channels=input_channels).to(device)
        elif current_model_type_arg == 'mlp':
            flattened_input_dim = input_size_hw[0] * input_size_hw[1] * input_channels
            model = MLP(input_dim=flattened_input_dim, hidden_dims=args.mlp_hidden_dims,
                        output_dim=num_classes).to(device)
        elif current_model_type_arg == 'fourier':
            model = FourierNet(input_size_hw=input_size_hw, input_channels=input_channels,
                               crop_fraction=args.fourier_crop_fraction, mlp_hidden_dims=args.fourier_mlp_hidden_dims,
                               output_dim=num_classes).to(device)
        else:
            raise ValueError(f"Unknown model type: {current_model_type_arg}")

        print(model)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters for {current_model_type_arg}: {total_params:,}")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        train_losses_hist, test_losses_hist, test_accuracies_hist = [], [], []

        print(
            f"\n--- Starting Training for {current_model_type_arg} on {training_dataset_name} for {args.epochs} epochs ---")
        for epoch in range(args.epochs):
            model.train();
            epoch_train_loss = 0.0
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                if current_model_type_arg == 'mlp': images = images.view(images.size(0), -1)
                outputs = model(images)
                if isinstance(outputs, tuple): outputs = outputs[0]
                loss = criterion(outputs, labels)
                optimizer.zero_grad();
                loss.backward();
                optimizer.step()
                epoch_train_loss += loss.item()
                if (batch_idx + 1) % 50 == 0 or batch_idx == len(train_loader) - 1:
                    print(
                        f'Model [{current_model_type_arg}] Epoch [{epoch + 1}/{args.epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            avg_epoch_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else 0
            train_losses_hist.append(avg_epoch_train_loss)
            print(
                f'Model [{current_model_type_arg}] Epoch [{epoch + 1}/{args.epochs}] Avg Training Loss: {avg_epoch_train_loss:.4f}')

            model.eval();
            epoch_test_loss, correct_preds, total_samples_eval = 0.0, 0, 0
            if clean_test_loader:
                with torch.no_grad():
                    for images, labels in clean_test_loader:
                        images, labels = images.to(device), labels.to(device)
                        if current_model_type_arg == 'mlp': images = images.view(images.size(0), -1)
                        outputs = model(images)
                        if isinstance(outputs, tuple): outputs = outputs[0]
                        loss = criterion(outputs, labels);
                        epoch_test_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total_samples_eval += labels.size(0);
                        correct_preds += (predicted == labels).sum().item()
                avg_epoch_test_loss = epoch_test_loss / len(clean_test_loader) if len(clean_test_loader) > 0 else 0
                accuracy = 100 * correct_preds / total_samples_eval if total_samples_eval > 0 else 0
                test_losses_hist.append(avg_epoch_test_loss)
                test_accuracies_hist.append(accuracy)
                print(
                    f'Model [{current_model_type_arg}] Epoch [{epoch + 1}/{args.epochs}] Clean Test Loss: {avg_epoch_test_loss:.4f}, Clean Test Accuracy: {accuracy:.2f}%')
            else:
                print(f"Warning: Clean test loader not available for epoch {epoch + 1} evaluation.")

        print(f'--- Finished Training for {current_model_type_arg} ---')

        print(
            f"\n--- Generating Standard Plots for {current_model_type_arg} (based on clean data training/evaluation) ---")
        plot_training_metrics(train_losses_hist, test_losses_hist, test_accuracies_hist, args.epochs,
                              save_dir=model_save_directory)

        if clean_test_loader and len(clean_test_loader) > 0:
            print(f"Extracting penultimate features for t-SNE ({current_model_type_arg} model on clean test data)...")
            features, true_labels = get_penultimate_features(model, clean_test_loader, device, current_model_type_arg,
                                                             num_samples=args.tsne_samples)
            if features.shape[0] > 0:
                plot_feature_space_projection(features, true_labels, model_specific_tag_for_legend,
                                              save_dir=model_save_directory,
                                              dataset_name_for_plot=f"{training_dataset_name} (Clean Test)")

        if current_model_type_arg in ['waveron', 'waveron_concat', 'single_layer_waveron_concat']:
            if total_params < 2_000_000:
                print("Generating Waveron Layer 1 filter visualizations...")
                plot_waveron_filters(model, save_dir=model_save_directory)  # kernel_size is implicitly in model
            if clean_test_loader and len(clean_test_loader) > 0:
                print("Generating Waveron comprehensive feature visualizations (on clean test data)...")
                plot_waveron_features(model, clean_test_loader, device, num_samples=min(2, args.batch_size),
                                      save_dir=model_save_directory,
                                      dataset_name_for_plot=f"{training_dataset_name} (Clean Test)")
                print("Calculating and plotting Waveron average sub-band energies (on clean test data)...")
                calculate_and_plot_average_subband_energies(model, clean_test_loader, device,
                                                            save_dir=model_save_directory,
                                                            dataset_name_for_plot=f"{training_dataset_name} (Clean Test)",
                                                            model_type_for_plot=current_model_type_arg)
        elif current_model_type_arg == 'fourier':
            if clean_test_loader and len(clean_test_loader) > 0:
                print("Generating Fourier Network analysis plots (on clean test data)...")
                plot_fourier_network_analysis(model, clean_test_loader, device,
                                              save_dir=model_save_directory,
                                              dataset_name_for_plot=f"{training_dataset_name} (Clean Test)")

        if args.eval_noise_levels:
            print(f"\n--- Evaluating Robustness to Noise for model: {current_model_type_arg} ---")
            current_model_noise_accuracies = []
            for noise_level in args.eval_noise_levels:
                print(f"Evaluating {current_model_type_arg} on MNIST with noise_std = {noise_level}...")
                _, noisy_test_loader, _, _, _ = get_dataset(
                    'mnist_noisy', batch_size=args.batch_size, noise_std=noise_level,
                    data_root=args.data_root, target_img_size=(28, 28)
                )
                if noisy_test_loader is None:
                    print(f"Could not load noisy test set for std={noise_level}. Skipping.");
                    current_model_noise_accuracies.append(0);
                    continue
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
                    f"Accuracy for {current_model_type_arg} on noisy MNIST (std={noise_level}): {accuracy_noisy:.2f}%")
            all_models_noise_evaluation_accuracies[model_specific_tag_for_legend] = current_model_noise_accuracies

        if args.eval_bg_rand_thresholds:
            print(f"\n--- Evaluating Robustness to Background Randomization for model: {current_model_type_arg} ---")
            current_model_bg_rand_accuracies = []
            for bg_thresh in args.eval_bg_rand_thresholds:
                print(f"Evaluating {current_model_type_arg} on MNIST with bg_rand_threshold = {bg_thresh}...")
                _, bg_rand_test_loader, _, _, _ = get_dataset(
                    'mnist_bg_rand', batch_size=args.batch_size, bg_rand_threshold=bg_thresh,
                    data_root=args.data_root, target_img_size=(28, 28)
                )
                if bg_rand_test_loader is None:
                    print(f"Could not load bg_rand test set for threshold={bg_thresh}. Skipping.");
                    current_model_bg_rand_accuracies.append(0);
                    continue
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
                    f"Accuracy for {current_model_type_arg} on MNIST with bg_rand_thresh={bg_thresh}: {accuracy_bg_rand:.2f}%")
            all_models_bg_rand_evaluation_accuracies[model_specific_tag_for_legend] = current_model_bg_rand_accuracies
    # End of loop for model_types

    if args.eval_noise_levels and all_models_noise_evaluation_accuracies:
        print("\n--- Generating Combined Accuracy vs. Noise Plot for all models ---")
        plot_accuracy_vs_noise(
            noise_levels_list=args.eval_noise_levels,
            all_models_accuracies_dict=all_models_noise_evaluation_accuracies,
            save_dir=experiment_base_dir,
            dataset_name_for_plot=args.dataset
        )

    if args.eval_bg_rand_thresholds and all_models_bg_rand_evaluation_accuracies:
        print("\n--- Generating Combined Accuracy vs. Background Threshold Plot for all models ---")
        plot_accuracy_vs_bg_threshold(
            bg_thresholds_list=args.eval_bg_rand_thresholds,
            all_models_accuracies_dict=all_models_bg_rand_evaluation_accuracies,
            save_dir=experiment_base_dir,
            dataset_name_for_plot=args.dataset
        )

    plt.show()
    print(f"--- Experiment Finished. All results saved under {experiment_base_dir} ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Waveron Network Research Experiment")

    parser.add_argument('--model_types', type=str, nargs='+', default=['waveron', 'mlp', 'fourier'],
                        choices=['waveron', 'waveron_concat', 'single_layer_waveron_concat', 'mlp', 'fourier'],
                        help='List of model types to train and evaluate')
    parser.add_argument('--dataset', type=str, default='mnist_noisy',
                        help='Base dataset for training (or context if --eval_... is used, e.g. "mnist" for clean training)')
    parser.add_argument('--data_root', type=str, default='./data', help='Root directory for datasets')
    parser.add_argument('--target_img_size', type=int, default=64, help='Target image H/W (MNIST uses 28)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--tsne_samples', type=int, default=1000, help='Number of samples for t-SNE plot')

    parser.add_argument('--noise_std', type=float, default=0.2,
                        help='Std dev for Gaussian noise IF training on mnist_noisy variants. Not used if --eval_noise_levels.')
    parser.add_argument('--bg_rand_threshold', type=float, default=0.1,
                        help='Threshold for AddRandomBackground transform IF training on bg_rand variants.')

    parser.add_argument('--eval_noise_levels', type=float, nargs='*', default=[0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        help='List of noise std deviations for robustness evaluation. Trains on clean MNIST.')
    parser.add_argument('--eval_bg_rand_thresholds', type=float, nargs='*', default=None,
                        help='List of bg_rand thresholds for robustness evaluation. Trains on clean MNIST.')

    # Waveron specific arguments
    parser.add_argument('--wavelet', type=str, default='sym4', help='Wavelet type for Waveron network')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='Conv kernel size in Waverons (Used for conv-based Waveron)')
    parser.add_argument('--waveron_channels', type=int, default=8,
                        help='Output channels for Waveron convolutions / or hint for mask-based Waveron channel depth')

    parser.add_argument('--mlp_hidden_dims', type=int, nargs='+', default=[32, 16],
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