import numpy as np
from scipy.spatial.distance import cdist
from skimage import io, exposure, measure, filters, color
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle

from sklearn.metrics import confusion_matrix

PATH = 'saved_images'
MIN_AREA_THRESHOLD = 90


def read_and_visualize_image(character, image_path, display_plots):
    image = io.imread(image_path)  # Read image
    if display_plots:
        plt.figure()  # Create a new figure
        io.imshow(image)  # Display image on the figure
        plt.title('Original Image')  # Set title for the image
        plt.savefig(f'{PATH}/{character}_original_image.png')  # Save image
        plt.close()  # Close the figure
    return image


def visualize_histogram(character, image, display_plots):
    histogram = exposure.histogram(image)
    if display_plots:
        plt.figure()
        plt.bar(histogram[1], histogram[0])
        plt.title(f'{character} - Histogram')
        plt.savefig(f'{PATH}/{character}_histogram.png')
        plt.close()


def binarize_image(character, image, threshold, display_plots):
    binarized_image = (image < threshold).astype(np.double)
    if display_plots:
        plt.figure()
        plt.imshow(binarized_image, cmap='gray')
        plt.title(f'{character} - Binarized Image')
        plt.savefig(f'{PATH}/{character}_binarized_image.png')
        plt.close()
    return binarized_image


def label_and_display_components(character, binarized_image, display_plots):
    labeled_image = measure.label(binarized_image, background=0)
    if display_plots:
        plt.figure()
        plt.imshow(labeled_image, cmap='nipy_spectral')
        plt.title(f'{character} - Labeled Image')
        plt.savefig(f'{PATH}/{character}_labeled_image.png')
        plt.close()
    return labeled_image


def extract_features_for_each_character(character, binarized_image, image_label, original_image=None, display_plots=True):
    regions = measure.regionprops(image_label)
    features_list = []

    image_with_boxes = None

    if original_image is not None:
        image_with_boxes = color.label2rgb(image_label, image=original_image, bg_label=0)

    fig, ax = plt.subplots() if display_plots else (None, None)

    if image_with_boxes is not None and display_plots:
        ax.imshow(image_with_boxes)

    for properties in regions:
        if properties.area < MIN_AREA_THRESHOLD:
            continue

        min_row, min_col, max_row, max_col = properties.bbox
        roi = binarized_image[min_row:max_row, min_col:max_col]
        m = measure.moments(roi)
        cr = m[0, 1] / m[0, 0]
        cc = m[1, 0] / m[0, 0]
        mu = measure.moments_central(roi, center=(cr, cc))
        nu = measure.moments_normalized(mu)
        hu = measure.moments_hu(nu)
        labeled_features = np.insert(hu, 0, ord(character))
        features_list.append(labeled_features)

        if display_plots:
            rect = Rectangle((min_col, min_row), max_col - min_col, max_row - min_row,
                             edgecolor='red', facecolor='none', linewidth=2)
            ax.add_patch(rect)

    if display_plots:
        ax.set_title(f'{character} - Image with Bounding Boxes')
        ax.axis('off')
        plt.savefig(f'{PATH}/{character}_image_with_bounding_boxes.png')
        plt.close(fig)

    return features_list


def train_ocr_system(image_paths, display_plots=True):
    all_features = []
    features_database = {}

    for character, image_path in image_paths.items():
        print(f"Processing character '{character}'...")
        image = read_and_visualize_image(character, image_path, display_plots)
        visualize_histogram(character, image, display_plots)
        threshold_value = filters.threshold_otsu(image)
        binarized_image = binarize_image(character, image, threshold_value, display_plots)
        image_label = label_and_display_components(character, binarized_image, display_plots)
        features = extract_features_for_each_character(character, binarized_image, image_label,
                                                       original_image=image, display_plots=display_plots)

        # Add label to each feature vector
        labeled_features = [np.insert(feat, 0, ord(character)) for feat in features]
        features_database[character] = labeled_features
        all_features.extend(labeled_features)

    # Convert all_features to a numpy array for processing
    all_features = np.array(all_features)

    # Calculate mean and std excluding the label column
    mean = np.mean(all_features[:, 1:], axis=0)
    std = np.std(all_features[:, 1:], axis=0)

    # Normalize features excluding the label
    for character, features in features_database.items():
        normalized_features = [(feat[1:] - mean) / std for feat in features]
        # Re-add the labels after normalization
        features_database[character] = [np.insert(norm_feat, 0, feat[0]) for norm_feat, feat in zip(normalized_features, features)]

    # Save the features database and normalization parameters
    with open('features_database.pkl', 'wb') as f:
        pickle.dump(features_database, f)
        pickle.dump((mean, std), f)

    return features_database, mean, std


def evaluate_on_training_data(features_database, mean, std):
    # Convert the feature database to a numpy array and separate labels from features
    labels = []
    features = []
    for char, feats in features_database.items():
        for feat in feats:
            labels.append(feat[0])  # The label is the first element
            features.append(feat[1:])  # The rest are features

    features = np.array(features)

    # Normalize the features
    normalized_features = (features - mean) / std

    # Compute distances using cdist
    distances = cdist(normalized_features, normalized_features)

    # Find the second nearest neighbor for each feature vector
    nearest_neighbors = []
    for i, row in enumerate(distances):
        sorted_indices = np.argsort(row)
        for idx in sorted_indices:
            if idx != i:  # Exclude self-match
                nearest_neighbors.append(labels[idx])
                break

    # Convert labels back to characters
    nearest_neighbors = [chr(int(label)) for label in nearest_neighbors]
    true_labels = [chr(int(label)) for label in labels]

    return nearest_neighbors, true_labels



if __name__ == '__main__':
    image_paths = {
        'a': 'images/a.bmp',
        'd': 'images/d.bmp',
        'm': 'images/m.bmp',
        'n': 'images/n.bmp',
        'o': 'images/o.bmp',
        'p': 'images/p.bmp',
        'q': 'images/q.bmp',
        'r': 'images/r.bmp',
        'u': 'images/u.bmp',
        'w': 'images/w.bmp',
    }

    features_database, mean, std = train_ocr_system(image_paths, display_plots=False)
    nearest_neighbors, true_labels = evaluate_on_training_data(features_database, mean, std)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(true_labels, nearest_neighbors)
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]


