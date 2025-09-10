import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, label
from scipy.spatial import KDTree
from scipy.stats import mode

from skimage import measure, img_as_ubyte, morphology, filters
from skimage.color import rgb2hed, hed2rgb
from skimage.measure import find_contours

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import cv2
import os
import sqlite3
import json
import urllib.request

def load_data(url, mask_path, cell_mask_path):
    try:
        req = urllib.request.urlopen(url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        ihc_img = cv2.imdecode(arr, -1)
        org = cv2.cvtColor(ihc_img, cv2.COLOR_BGR2RGB)

        mask_data = np.load(mask_path)
        mask = mask_data['mask']
        mask = correct_mask(mask)

        # Display the NumPy array using Matplotlib without axis
        # plt.imshow(mask, cmap='viridis')  # Use an appropriate colormap
        # plt.axis('off')  # Turn off the axis
        #plt.show()

        # org = cv2.imread(image_path, 1)
        # org = cv2.cvtColor(org, cv2.COLOR_BGR2RGB)


        cell_mask_data = np.load(cell_mask_path)
        cell_mask = cell_mask_data['mask']
        return org, mask, cell_mask

    except Exception as e:
        print(f"Error loading data for {image_path}: {e}")
        return None, None, None


def correct_class(mask, class_value):
    class_mask = (mask == class_value)
    labeled_mask, num_features = label(class_mask)

    sizes = np.bincount(labeled_mask.ravel())
    sizes[0] = 0  # Hintergrundkomponente ignorieren

    largest_component_size = sizes.max()
    min_size_threshold = largest_component_size / 8

    new_class_mask = np.zeros_like(mask)

    for component_label, size in enumerate(sizes):
        if size >= min_size_threshold:
            new_class_mask[labeled_mask == component_label] = class_value

    return new_class_mask

def correct_mask(mask):
    mask = mask.astype(np.int32)

    # class 1 (Dermis)
    corrected_class_1_mask = correct_class(mask, 1)

    # class 2 (Epidermis)
    corrected_class_2_mask = correct_class(mask, 2)

    new_mask = mask.copy()

    # Dermis
    new_mask[mask == 1] = 0
    new_mask += corrected_class_1_mask.astype(np.int32)

    # Epidermis
    new_mask[mask == 2] = 0
    new_mask += corrected_class_2_mask.astype(np.int32)

    return new_mask.astype(mask.dtype)

def compute_distance_transform(mask):
    epidermis_mask = (mask == 2)
    dermis_mask = (mask == 1)

    distance = distance_transform_edt(~dermis_mask)
    distance[~epidermis_mask] = 0

    return distance, epidermis_mask

def calculate_cell_diameter(cell_mask):
    labels = measure.label(cell_mask)
    props = measure.regionprops(labels)

    cell_diameters = [prop.equivalent_diameter for prop in props]
    mean_cell_diameter = np.mean(cell_diameters)

    print("Mean cell diameter:", mean_cell_diameter)

    return mean_cell_diameter

def divide_into_layers(distance, epidermis_mask, mean_cell_diameter):
    num_layers = int(np.ceil(distance.max() / mean_cell_diameter))
    layered_epidermis = np.zeros_like(distance)
    layer_intervals = np.linspace(0, distance.max(), num_layers + 1)

    for i in range(num_layers):
        layer_start = layer_intervals[i]
        layer_end = layer_intervals[i + 1]
        mask_layer = (distance >= layer_start) & (distance < layer_end)
        layered_epidermis[mask_layer & epidermis_mask] = i + 1

    print("Number of layers:", num_layers)

    return layered_epidermis, num_layers

# def calculate_layer_thickness_and_std(distance, num_layers):
#     layer_thicknesses = np.zeros(num_layers)
#     layer_stds = np.zeros(num_layers)
#
#     layer_intervals = np.linspace(0, distance.max(), num_layers + 1)
#     layer_lengths = []
#
#     for i in range(1, num_layers):
#         layer_start = layer_intervals[i]
#         layer_end = layer_intervals[i + 1]
#         mask_layer = (distance >= layer_start) & (distance < layer_end)
#         layer_length = len(distance[mask_layer])
#         layer_lengths.append(layer_length)
#
#         # Thickness for each layer
#         layer_thickness = distance[mask_layer] / i
#         mean_thickness = np.mean(layer_thickness)
#         std_thickness = np.std(layer_thickness)
#
#         # Calculate the scaled mean thickness and std thickness
#         scaling_factor = layer_length / max(layer_lengths) if layer_lengths else 1
#         print(scaling_factor)
#         layer_thicknesses[i] = mean_thickness * scaling_factor
#         layer_stds[i] = std_thickness * scaling_factor
#
#     # Sum the scaled layer thicknesses and stds
#     total_weighted_thickness = np.sum(layer_thicknesses)
#     total_weighted_std = np.sum(layer_stds)
#     print("Total Weighted Thickness:", total_weighted_thickness)
#     print("Total Weighted Std:", total_weighted_std)
#
#
#     return layer_thicknesses, layer_stds, layer_lengths


def calculate_layer_thickness_and_std(distance, num_layers):
    layer_thicknesses = np.zeros(num_layers)
    layer_stds = np.zeros(num_layers)

    layer_intervals = np.linspace(0, distance.max(), num_layers + 1)
    layer_lengths = []

    for i in range(1, num_layers):
        layer_start = layer_intervals[i]
        layer_end = layer_intervals[i + 1]
        mask_layer = (distance >= layer_start) & (distance < layer_end)
        layer_length = len(distance[mask_layer])
        layer_lengths.append(layer_length)

    # Calculate the maximum layer length for scaling
    max_layer_length = max(layer_lengths)

    for i in range(1, num_layers):
        layer_start = layer_intervals[i]
        layer_end = layer_intervals[i + 1]
        mask_layer = (distance >= layer_start) & (distance < layer_end)

        layer_length = layer_lengths[i - 1]

        # Thickness for each layer
        layer_thickness = distance[mask_layer] / i
        mean_thickness = np.mean(layer_thickness)
        std_thickness = np.std(layer_thickness)

        # Calculate the scaled mean thickness and std thickness
        scaling_factor = layer_length / max_layer_length if max_layer_length > 0 else 1
        layer_thicknesses[i] = mean_thickness * scaling_factor
        layer_stds[i] = std_thickness * scaling_factor

        print(scaling_factor)

    # Sum the scaled layer thicknesses and stds
    total_weighted_thickness = np.sum(layer_thicknesses)
    total_weighted_std = np.sum(layer_stds)
    print("Total Weighted Thickness:", total_weighted_thickness)
    print("Total Weighted Std:", total_weighted_std)

    return layer_thicknesses, layer_stds, layer_lengths


# def calculate_layer_thickness_and_std(distance, num_layers):
#     layer_thicknesses = np.zeros(num_layers)
#     layer_stds = np.zeros(num_layers)
#
#     # Calculate the intervals for the layers
#     layer_intervals = np.linspace(0, distance.max(), num_layers + 1)
#     layer_lengths = []
#
#     # Step 1: Calculate the lengths of each layer
#     for i in range(1, num_layers):
#         layer_start = layer_intervals[i]
#         layer_end = layer_intervals[i + 1]
#         mask_layer = (distance >= layer_start) & (distance < layer_end)
#         layer_length = len(distance[mask_layer])
#         layer_lengths.append(layer_length)
#
#     # Calculate the total average length and use half of it for scaling
#     total_average_length = sum(layer_lengths) / num_layers
#     half_total_average_length = total_average_length / 2
#
#     # Step 2: Calculate the thickness and std for each layer, with scaling
#     for i in range(1, num_layers):
#         layer_start = layer_intervals[i]
#         layer_end = layer_intervals[i + 1]
#         mask_layer = (distance >= layer_start) & (distance < layer_end)
#         layer_length = layer_lengths[i - 1]
#
#         # Thickness for each layer
#         layer_thickness = distance[mask_layer] / i
#         mean_thickness = np.mean(layer_thickness)
#         std_thickness = np.std(layer_thickness)
#
#         # Calculate the scaled mean thickness and std thickness
#         scaling_factor = layer_length / half_total_average_length if half_total_average_length > 0 else 1
#         layer_thicknesses[i] = mean_thickness * scaling_factor
#         layer_stds[i] = std_thickness * scaling_factor
#
#         print(f"Layer {i}: Scaling factor = {scaling_factor}")
#
#     # Sum the scaled layer thicknesses and stds
#     total_weighted_thickness = np.sum(layer_thicknesses)
#     total_weighted_std = np.sum(layer_stds)
#     print("Total Weighted Thickness:", total_weighted_thickness)
#     print("Total Weighted Std:", total_weighted_std)
#
#     return layer_thicknesses, layer_stds, layer_lengths

def color_separate(ihc_rgb):
    ihc_hed = rgb2hed(ihc_rgb)
    null = np.zeros_like(ihc_hed[:, :, 0])
    ihc_h = img_as_ubyte(hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1)))
    ihc_d = img_as_ubyte(hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1)))

    ihc_h = np.invert(ihc_h[:, :, 2])
    ihc_d = np.invert(ihc_d[:, :, 2])

    return ihc_h, ihc_d

'''Stop at 80% of num_layers'''
# def calculate_intensity_per_layer(layered_epidermis, channel, num_layers, ihc_h=False):
#     intensities = []
#
#     # Stopping point for ihc_h
#     stopping_point = int(0.80 * num_layers) if ihc_h else num_layers
#
#     for i in range(1, num_layers + 1):
#         if i <= stopping_point:
#             layer_mask = (layered_epidermis == i)
#             mean_intensity = np.mean(channel[layer_mask])
#             intensities.append(mean_intensity)
#         else:
#             # Intensity for remaining layers
#             intensities.append(intensities[-1])
#
#     return intensities

def calculate_intensity_per_layer(layered_epidermis, channel, num_layers, ihc_h=False):
    intensities = []

    for i in range(1, num_layers + 1):
        layer_mask = (layered_epidermis == i)
        mean_intensity = np.mean(channel[layer_mask])
        intensities.append(mean_intensity)

    return intensities

def calculate_intensity_gradients(intensities):
    gradients = np.diff(intensities)
    return gradients

def perform_kmeans_clustering(data):
    data = np.array(data).reshape(-1, 1)
    max_k = max(2, len(data) // 2)

    silhouette_avg = []
    K = range(2, max_k + 1)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        cluster_labels = kmeans.labels_
        silhouette_avg.append(silhouette_score(data, cluster_labels))

    optimal_k = K[np.argmax(silhouette_avg)]
    print(optimal_k, 'Number of Clusters')

    num_clusters = optimal_k

    # Fit KMeans with optimal number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(data)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_


    plt.figure(figsize=(8, 6))
    plt.scatter(np.arange(1, len(data) + 1), data, c=cluster_labels, cmap='viridis', edgecolors='k')
    #plt.scatter(np.arange(1, len(cluster_centers) + 1), cluster_centers, marker='x', c='red', s=100, label='Cluster Centers')
    plt.title('K-Means Clustering of Gradients')
    plt.xlabel('Data Points/Layers')
    plt.ylabel('Gradient')
    plt.xticks(range(1, len(data) + 1))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    return cluster_labels, cluster_centers



def plot_layer_thickness(thicknesses, stds, lengths):
    num_layers = len(thicknesses)
    layers = np.arange(1, num_layers + 1)

    # Creating subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
    ax1.set_title('Thickness per Layer')

    # Plot mean thicknesses with error bars for standard deviations
    #ax1.set_xlabel('Layer')
    ax1.set_ylabel('Mean Thickness', color='tab:blue')
    ax1.errorbar(layers[1:], thicknesses[1:], yerr=stds[1:], fmt='o-', color='tab:blue', label='Scaled Mean Thickness')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xticks(range(2, num_layers + 1))

    # # Plot standard deviations
    # ax2.set_ylabel('Std Deviation', color='tab:red')
    # ax2.plot(layers[1:], stds[1:], 's--', color='tab:red', label='Scaled Std Deviation')
    # ax2.tick_params(axis='y', labelcolor='tab:red')
    # ax2.set_xticks(range(2, num_layers + 1))

    # Plot lengths as a bar chart
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Number of Elements', color='gray')
    ax2.bar(layers[1:], lengths, color='gray', label='Number of Elements')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_xticks(range(2, num_layers + 1))

    fig.tight_layout()
    plt.suptitle('Layer Information', y=1.02)  # Title above the subplots


def plot_original_D_H_intensities(org, ihc_d, ihc_h, d_intensities, h_intensities, num_layers):
    fig, axes = plt.subplots(2, 3, figsize=(15, 5))

    # Original image
    ax = axes[0, 0]
    ax.imshow(org)
    ax.axis('off')
    ax.set_title('Original Image')

    # DAB channel
    ax = axes[0, 1]
    ax.imshow(ihc_d, cmap='gray')
    ax.axis('off')
    ax.set_title('DAB Channel')

    # Plot DAB intensities
    ax = axes[0, 2]
    ax.plot(range(1, num_layers + 1), d_intensities, marker='o')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean DAB Intensity')
    ax.set_xticks(range(1, num_layers + 1))
    ax.set_title('Mean DAB Intensity')
    ax.grid(True)

    # Original image (duplicate)
    ax = axes[1, 0]
    ax.imshow(org)
    ax.axis('off')
    ax.set_title('Original Image')

    # H channel
    ax = axes[1, 1]
    ax.imshow(ihc_h, cmap='gray')
    ax.axis('off')
    ax.set_title('H Channel')

    # Plot H intensities
    ax = axes[1, 2]
    ax.plot(range(1, num_layers + 1), h_intensities, marker='o')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean H Intensity')
    ax.set_xticks(range(1, num_layers + 1))
    ax.set_title('Mean H Intensity')
    ax.grid(True)

    plt.tight_layout()

def plot_D_H_ratio_and_layerd_epidermis(org, layered_epidermis, epidermis_mask, ratio_intensities, num_layers):
    # Contour of the epidermis
    contours = measure.find_contours(epidermis_mask, 0.5)

    # Mean intensity and ratio plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot DAB/H ratio intensities
    ax = axes[0]
    ax.plot(range(1, num_layers + 1), ratio_intensities, marker='o')
    ax.set_xlabel('Layer')
    ax.set_ylabel('DAB/H Intensity')
    ax.set_xticks(range(1, num_layers + 1))
    ax.set_title('DAB/H Intensity')
    ax.grid(True)

    # Layered epidermis overlaid on original image
    ax = axes[1]
    ax.imshow(org)
    ax.imshow(layered_epidermis, alpha=0.5, cmap='jet')
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], 'w-', linewidth=2)
    ax.axis('off')
    ax.set_title('Layered Epidermis')

    plt.tight_layout()



def plot_org_mask_layerd_epidermis(org, mask, layered_epidermis, epidermis_mask, gene_name, filename):
    contours = measure.find_contours(epidermis_mask, 0.5)

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle(filename, fontsize=16)

    ax = axes[0]
    ax.imshow(org, cmap='gray' if len(org.shape) == 2 else None)
    ax.axis('off')
    ax.set_title(gene_name)

    ax = axes[1]
    ax.imshow(mask)
    ax.axis('off')
    ax.set_title('Mask')

    ax = axes[2]
    ax.imshow(org, cmap='gray' if len(org.shape) == 2 else None)
    ax.imshow(layered_epidermis, alpha=0.5, cmap='jet')
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], 'w-', linewidth=2)
    ax.axis('off')
    ax.set_title('Layered Epidermis')

    plt.tight_layout()
   # plt.show()


def plot_gradient_intensity(ihc_d_gradients, ihc_h_gradients, ratio_gradients, num_layers):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_layers), ihc_h_gradients, marker='o', label='Hematoxylin Gradient')
    plt.plot(range(1, num_layers), ihc_d_gradients, marker='o', label='DAB Gradient')
    plt.plot(range(1, num_layers), ratio_gradients, marker='o', label='Ratio Gradient')

    plt.xlabel('Layer')
    plt.ylabel('Gradient of Intensity')
    plt.title('Gradient of Intensity over Layer')
    plt.xticks(range(1, num_layers + 1))

    plt.legend()
    plt.grid(True)

def process_image(filename, main_id, url, gene_name):
    mask_path = f'data/masks_IHC/{filename}.npz'
    cell_mask_path = f'data/masks_IHC_cells/{filename}.npz'

    try:
        '''Start Calculating'''
        org, mask, cell_mask = load_data(url, mask_path, cell_mask_path)
        mean_cell_diameter = calculate_cell_diameter(cell_mask)

        distance, epidermis_mask = compute_distance_transform(mask)
        layered_epidermis, num_layers = divide_into_layers(distance, epidermis_mask, mean_cell_diameter)

        #layer_thicknesses, layer_stds, layer_lengths = calculate_layer_thickness_and_std(distance, num_layers)

        ihc_h, ihc_d = color_separate(org)

        d_intensities = calculate_intensity_per_layer(layered_epidermis, ihc_d, num_layers, ihc_h=False)
        h_intensities = calculate_intensity_per_layer(layered_epidermis, ihc_h, num_layers, ihc_h=True)

        ratio_intensities = np.array(d_intensities) / np.array(h_intensities)

        plt.figure()  # Start a new figure for each plot

        plot_org_mask_layerd_epidermis(org, mask, layered_epidermis, epidermis_mask, gene_name, filename)
        #plot_gradient_intensity(ihc_d_gradients, ihc_h_gradients, ratio_gradients, num_layers)
       # plot_D_H_ratio_and_layerd_epidermis(org, layered_epidermis, epidermis_mask, ratio_intensities,
        #                                    num_layers)
       # plot_original_D_H_intensities(org, ihc_d, ihc_h, d_intensities, h_intensities, num_layers)
        #plot_layer_thickness(thicknesses, stds, lengths)

        plt.savefig(f'{filename}.jpg', dpi=300, facecolor='w')  # Ensure white background

        #plt.show()


        '''Save in Database'''
        conn = sqlite3.connect('FINAL.db')
        cursor = conn.cursor()

        # Create the table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_data (
            mainID TEXT PRIMARY KEY,
            mean_cell_diameter REAL,
            num_layers INTEGER,
            d_intensities TEXT,
            h_intensities TEXT,
            ratio_intensities TEXT
        )
        ''')

        d_intensities_json = json.dumps(d_intensities)
        h_intensities_json = json.dumps(h_intensities)
        ratio_intensities_json = json.dumps(ratio_intensities.tolist())  # Convert numpy array to list before JSON

        # Check if mainID already exists
        cursor.execute("SELECT COUNT(*) FROM processed_data WHERE mainID = ?", (main_id,))
        exists = cursor.fetchone()[0]

        if exists:
            cursor.execute('''
                    UPDATE processed_data
                    SET mean_cell_diameter = ?, num_layers = ?, d_intensities = ?, h_intensities = ?, ratio_intensities = ?
                    WHERE mainID = ?
                ''', (
            mean_cell_diameter, num_layers, d_intensities_json, h_intensities_json, ratio_intensities_json, main_id))
            print(f"Updated record with mainID: {main_id}")
        else:
            cursor.execute('''
                    INSERT INTO processed_data (
                        mainID, mean_cell_diameter, num_layers, d_intensities, h_intensities, ratio_intensities
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
            main_id, mean_cell_diameter, num_layers, d_intensities_json, h_intensities_json, ratio_intensities_json))
            print(f"Inserted new record with mainID: {main_id}")


        conn.commit()
        conn.close()

    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        return filename

    return None


def save_last_processed_id(id_value, last_processed_id, base_filename):
    filename = f"{base_filename}_{id_value}.txt"
    with open(filename, 'w') as file:
        file.write(str(last_processed_id))

def load_last_processed_id(base_filename, default_id):
    filename = f"{base_filename}_{default_id}.txt"
    try:
        with open(filename, 'r') as file:
            last_processed_id = int(file.read().strip())
            return last_processed_id
    except FileNotFoundError:
        return 0

def main(id_value=0):
    conn = sqlite3.connect('FINAL.db')
    cursor = conn.cursor()

    base_filename = 'last_processed_id'

    last_processed_id = load_last_processed_id(base_filename, id_value)

    name_filter = "TP63"  # Replace with the name you want to filter by
    cursor.execute("SELECT mainID, url FROM elGenes WHERE name = ? ORDER BY mainID ASC", (name_filter,))

    data = cursor.fetchall()

    failed_files = []

    start_processing = False if last_processed_id else True

    for row in data:
        main_id = row[0]
        url = row[1]

        if not start_processing:
            if main_id == last_processed_id:
                start_processing = True
            continue

        mask_filename = url.split('/')[-2] + '-' + url.split('/')[-1]
        mask_filename = mask_filename.split('.')[0]

        print(f"MainID: {main_id}, Mask Filename: {mask_filename}")
        failed_filename = process_image(mask_filename, main_id, url, name_filter)

        if not failed_filename:
            save_last_processed_id(id_value, main_id, base_filename)

        if failed_filename:
            failed_files.append((main_id, failed_filename))

    conn.close()


    for main_id, failed_filename in failed_files:
        print(f"Processing failed for mainID {main_id} with file {failed_filename}")

    with open("failed_files.txt", "w") as file:
        if failed_files:
            file.write("Failed files:\n")
            for f in failed_files:
                file.write(f + "\n")

if __name__ == '__main__':
    import sys

    id_value = 10000
    if len(sys.argv) > 1:
        try:
            id_value = int(sys.argv[1])
            if id_value not in {1, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000}:
                raise ValueError("Invalid ID value")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    main(id_value)