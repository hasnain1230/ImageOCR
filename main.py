import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure, measure, filters, color
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle

PATH = 'saved_images'
MIN_AREA_THRESHOLD = 90


def read_and_visualize_image(character, image_path):
    image = io.imread(image_path)  # Read image
    print(image.shape)  # (height, width)
    plt.figure()  # Create a new figure
    io.imshow(image)  # Display image on the figure
    plt.title('Original Image')  # Set title for the image
    plt.savefig(f'{PATH}/{character}_original_image.png')  # Save image
    plt.close()  # Close the figure
    return image


def visualize_histogram(character, image):
    histogram = exposure.histogram(image)
    plt.figure()
    plt.bar(histogram[1], histogram[0])
    plt.title(f'{character} - Histogram')
    plt.savefig(f'{PATH}/{character}_histogram.png')
    plt.close()


def binarize_image(character, image, threshold):
    binarized_image = (image < threshold).astype(np.double)
    plt.figure()
    plt.imshow(binarized_image, cmap='gray')
    plt.title(f'{character} - Binarized Image')
    plt.savefig(f'{PATH}/{character}_binarized_image.png')
    plt.close()
    return binarized_image

def label_and_display_components(character, binarized_image):
    labeled_image = measure.label(binarized_image, background=0)
    plt.figure()
    plt.imshow(labeled_image, cmap='nipy_spectral')  # Display labeled image... cmap='nipy_spectral' is for better visualization
    plt.title(f'{character} - Labeled Image')
    plt.savefig(f'{PATH}/{character}_labeled_image.png')
    plt.close()
    print(np.amax(labeled_image))
    return labeled_image


def extract_features_for_each_character(character, binarized_image, image_label, original_image=None):
    regions = measure.regionprops(image_label)
    features_list = []

    image_with_boxes = None

    if original_image is not None:
        image_with_boxes = color.label2rgb(image_label, image=original_image, bg_label=0)

    fig, ax = plt.subplots()

    if image_with_boxes is not None:
        ax.imshow(image_with_boxes)

    for properties in regions:
        # Filter out small components
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

        # Draw bounding box
        rect = Rectangle((min_col, min_row), max_col - min_col, max_row - min_row,
                         edgecolor='red', facecolor='none', linewidth=2)

        ax.add_patch(rect)

    ax.set_title(f'{character} - Image with Bounding Boxes')
    ax.axis('off')
    plt.savefig(f'{PATH}/{character}_image_with_bounding_boxes.png')
    plt.close(fig)

    return features_list


# Main code to process training images
def train_ocr_system(image_paths):
    all_features = []
    features_database = {}

    for character, image_path in image_paths.items():
        image = read_and_visualize_image(character, image_path) # Read and visualize image of each character
        visualize_histogram(character, image) # Visualize histogram of each character
        threshold_value = filters.threshold_otsu(image) # Calculate threshold value using Otsu's method
        binarized_image = binarize_image(character, image, threshold_value) # Binarize image
        image_label = label_and_display_components(character, binarized_image)  # Label and display connected components
        features = extract_features_for_each_character(character, binarized_image, image_label, original_image=image)  # Extract features for each character
        features_database[character] = features
        all_features.extend(features)

    all_features = np.array(all_features)

    mean = np.mean(all_features, axis=0)
    std = np.std(all_features, axis=0)

    for character, features in features_database.items():
        features_database[character] = [(np.array(feat) - mean) / std for feat in features_database[character]]

    with open('features_database.pkl', 'wb') as f:
        pickle.dump(features_database, f)
        pickle.dump((mean, std), f)

    return features_database, mean, std



def read_and_binarize_test_image(image_path, threshold=200):
    test_image = io.imread(image_path)
    binarized_test_image = (test_image < threshold).astype(np.double)
    return binarized_test_image

def extract_and_classify_characters(test_image_path, features_database, mean, std):
    # Read and binarize test image
    test_image = io.imread(test_image_path)
    binarized_test_image = (test_image < 200).astype(np.double)  # Example threshold, adjust as needed
    image_label = label_and_display_components('test', binarized_test_image)
    test_features = extract_features_for_each_character("test", binarized_test_image, image_label)

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


# Assuming test_image_path is the path to your test image
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

features_database, mean, std = train_ocr_system(image_paths)

test_image_path = 'images/test.bmp'

recognized_characters = extract_and_classify_characters(test_image_path, features_database, mean, std)

print(recognized_characters)
print(len(recognized_characters))




