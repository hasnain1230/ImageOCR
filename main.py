from matplotlib.patches import Rectangle
from scipy.spatial.distance import cdist
from skimage import io, filters, measure, exposure, color
import matplotlib.pyplot as plt
import numpy as np
import pickle

PATH = 'saved_images'
MIN_AREA_THRESHOLD = 90

def binarize_image(character, image, threshold, display_plots=True):
    binarized_image = (image < threshold).astype(np.double)
    if display_plots:
        plt.figure()
        plt.imshow(binarized_image, cmap='gray')
        plt.title(f'{character} - Binarized Image')
        plt.savefig(f'{PATH}/{character}_binarized_image.png')
        plt.close()
    return binarized_image


def label_and_display_components(character, binarized_image, display_plots=True):
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
        features_list.append(hu)

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


def extract_and_classify_characters(test_image_path, features_database, mean, std):
    # Read and binarize test image
    test_image = io.imread(test_image_path)
    threshold_value = filters.threshold_otsu(test_image)
    binarized_test_image = binarize_image('test', test_image, threshold_value, display_plots=False)
    image_label = label_and_display_components('test', binarized_test_image, display_plots=False)
    test_features = extract_features_for_each_character("test", binarized_test_image, image_label, test_image, display_plots=False)

    # Normalize test features
    normalized_test_features = (np.array(test_features) - mean) / std

    # Combine all training features into one array for vectorized distance calculation
    all_training_features = []
    character_labels = []
    for character, training_features in features_database.items():
        all_training_features.extend(training_features)
        character_labels.extend([character] * len(training_features))

    all_training_features = np.array(all_training_features)

    # Compute distances between all pairs of test and training features
    distances = cdist(normalized_test_features, all_training_features)

    # Find the nearest training feature for each test feature
    recognized_characters = []
    for i in range(len(test_features)):
        nearest_index = np.argmin(distances[i])
        recognized_characters.append(character_labels[nearest_index])

    return recognized_characters


def load_trained_ocr_system():
    with open('features_database.pkl', 'rb') as f:
        features_database = pickle.load(f)
        mean, std = pickle.load(f)

    return features_database, mean, std


features_database, mean, std = load_trained_ocr_system()

test_image_path = 'images/test.bmp'

recognized_characters = extract_and_classify_characters(test_image_path, features_database, mean, std)

print(recognized_characters)

# EVALUATION..............................................
pkl_file = open('test_gt_py3.pkl', 'rb')
mydict = pickle.load(pkl_file)
pkl_file.close()
classes = mydict[b'classes']
locations = mydict[b'locations']  # locations of the characters in the image

