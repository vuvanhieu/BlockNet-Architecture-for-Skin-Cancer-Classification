# System and OS libraries
import os

# Visualization
import matplotlib.pyplot as plt

# TensorFlow and Keras for deep learning
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Image processing libraries
from PIL import Image
from skimage.feature import hog, local_binary_pattern
from skimage.transform import resize

# Numeric and scientific computing
import numpy as np
from math import ceil
from itertools import cycle
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray, rgb2hsv

import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
    
from tensorflow.keras.applications import (
    VGG16, VGG19, ResNet50, ResNet101, ResNet152, InceptionV3, MobileNet,
    EfficientNetB0, EfficientNetB7, DenseNet121, DenseNet169, DenseNet201
)
from tensorflow.keras.models import Model
import vit_keras.vit as vit
from transformers import SwinForImageClassification, CLIPModel
import torch
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import cv2
from skimage.feature import hog, local_binary_pattern, greycomatrix, greycoprops
from scipy.stats import kurtosis, skew
import pywt
from skimage.feature import greycomatrix as graycomatrix, greycoprops
from skimage.measure import shannon_entropy

from skfuzzy import control as ctrl
import skfuzzy as fuzz

def determine_optimal_entropy_range_fuzzy(entropies):
    """
    Xác định khoảng entropy tối ưu dựa trên mô hình mờ.
    """
    filtered_entropies = [e for e in entropies if e > 0.1]  # Lọc bỏ entropy không ý nghĩa
    if len(filtered_entropies) == 0:
        raise ValueError("No valid entropy values above the threshold.")

    # Khởi tạo các biến mờ
    entropy = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'entropy')
    range_out = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'range')

    # Định nghĩa hàm thuộc
    entropy['low'] = fuzz.trimf(entropy.universe, [0, 0, 0.5])
    entropy['medium'] = fuzz.trimf(entropy.universe, [0.2, 0.5, 0.8])
    entropy['high'] = fuzz.trimf(entropy.universe, [0.5, 1.0, 1.0])

    range_out['lower'] = fuzz.trimf(range_out.universe, [0, 0, 0.5])
    range_out['upper'] = fuzz.trimf(range_out.universe, [0.5, 1.0, 1.0])

    # Tập luật mờ
    rules = [
        ctrl.Rule(entropy['low'], range_out['lower']),
        ctrl.Rule(entropy['medium'], range_out['lower']),
        ctrl.Rule(entropy['high'], range_out['upper']),
    ]

    # Hệ mờ
    entropy_ctrl = ctrl.ControlSystem(rules)
    entropy_sim = ctrl.ControlSystemSimulation(entropy_ctrl)

    # Tính toán giá trị lower và upper
    entropy_sim.input['entropy'] = np.mean(filtered_entropies)
    entropy_sim.compute()
    lower = entropy_sim.output['range']

    entropy_sim.input['entropy'] = np.max(filtered_entropies)
    entropy_sim.compute()
    upper = entropy_sim.output['range']

    return lower, upper

def check_entropy(feature_map):
    """
    Kiểm tra entropy của feature map.
    """
    entropy = shannon_entropy(feature_map)
    return entropy

def determine_optimal_entropy_range(entropies):
    """Xác định khoảng entropy tối ưu dựa trên phân phối entropy loại bỏ các giá trị không ý nghĩa."""
    filtered_entropies = [e for e in entropies if e > 0.1]  # Lọc bỏ các giá trị entropy thấp không ý nghĩa
    if len(filtered_entropies) == 0:
        raise ValueError("No valid entropy values above the threshold.")
    mean_entropy = np.mean(filtered_entropies)
    std_entropy = np.std(filtered_entropies)
    lower = max(np.min(filtered_entropies), mean_entropy - std_entropy)  # Ngưỡng dưới: Mean - Std Dev
    upper = min(np.max(filtered_entropies), mean_entropy + std_entropy)  # Ngưỡng trên: Mean + Std Dev
    return lower, upper

# Load SOTA models
def load_SOTA_models():
    # VGG16 - Fully connected layer fc2
    model_vgg16 = VGG16(include_top=True, weights="imagenet")
    vgg16_fc2 = Model(inputs=model_vgg16.input, outputs=model_vgg16.get_layer("fc2").output)

    # VGG19 - Fully connected layer fc2
    model_vgg19 = VGG19(include_top=True, weights="imagenet")
    vgg19_fc2 = Model(inputs=model_vgg19.input, outputs=model_vgg19.get_layer("fc2").output)

    # ResNet50, ResNet101, ResNet152 - Global Average Pooling (avg_pool)
    model_resnet50 = ResNet50(include_top=True, weights="imagenet")
    resnet50_avg_pool = Model(inputs=model_resnet50.input, outputs=model_resnet50.get_layer("avg_pool").output)

    model_resnet101 = ResNet101(include_top=True, weights="imagenet")
    resnet101_avg_pool = Model(inputs=model_resnet101.input, outputs=model_resnet101.get_layer("avg_pool").output)

    model_resnet152 = ResNet152(include_top=True, weights="imagenet")
    resnet152_avg_pool = Model(inputs=model_resnet152.input, outputs=model_resnet152.get_layer("avg_pool").output)

    # InceptionV3 - Global Average Pooling (avg_pool)
    model_inceptionv3 = InceptionV3(include_top=True, weights="imagenet")
    inceptionv3_avg_pool = Model(inputs=model_inceptionv3.input, outputs=model_inceptionv3.get_layer("avg_pool").output)

    # MobileNet - Global Average Pooling
    model_mobilenet = MobileNet(include_top=True, weights="imagenet")
    mobilenet_avg_pool = Model(inputs=model_mobilenet.input, outputs=model_mobilenet.get_layer("global_average_pooling2d").output)

    # EfficientNetB0 and EfficientNetB7 - Global Average Pooling
    model_efficientnetb0 = EfficientNetB0(include_top=True, weights="imagenet")
    efficientnetb0_avg_pool = Model(inputs=model_efficientnetb0.input, outputs=model_efficientnetb0.get_layer("avg_pool").output)

    model_efficientnetb7 = EfficientNetB7(include_top=True, weights="imagenet")
    efficientnetb7_avg_pool = Model(inputs=model_efficientnetb7.input, outputs=model_efficientnetb7.get_layer("avg_pool").output)

    # DenseNet121, DenseNet169, DenseNet201 - Global Average Pooling
    model_densenet121 = DenseNet121(include_top=True, weights="imagenet")
    densenet121_avg_pool = Model(inputs=model_densenet121.input, outputs=model_densenet121.get_layer("avg_pool").output)

    model_densenet169 = DenseNet169(include_top=True, weights="imagenet")
    densenet169_avg_pool = Model(inputs=model_densenet169.input, outputs=model_densenet169.get_layer("avg_pool").output)

    model_densenet201 = DenseNet201(include_top=True, weights="imagenet")
    densenet201_avg_pool = Model(inputs=model_densenet201.input, outputs=model_densenet201.get_layer("avg_pool").output)

    # Vision Transformer (ViT) - Adjusted
    vit_b16_model = vit.vit_b16(image_size=224, pretrained=True, include_top=True)
    vit_b16_output = Model(inputs=vit_b16_model.input, outputs=vit_b16_model.layers[-2].output)  # Remove top layer

    # Dictionary of all models
    models = {
        "VGG16": vgg16_fc2,
        "VGG19": vgg19_fc2,
        "ResNet50": resnet50_avg_pool,
        "ResNet101": resnet101_avg_pool,
        "ResNet152": resnet152_avg_pool,
        "InceptionV3": inceptionv3_avg_pool,
        "MobileNet": mobilenet_avg_pool,
        "EfficientNetB0": efficientnetb0_avg_pool,
        "EfficientNetB7": efficientnetb7_avg_pool,
        "DenseNet121": densenet121_avg_pool,
        "DenseNet169": densenet169_avg_pool,
        "DenseNet201": densenet201_avg_pool,
        "ViT_B16": vit_b16_output,
    }

    return models


def balance_dataset(train_dir, labels):
    """
    Optimize the dataset to balance the number of images per label by calculating the required multipliers.

    Args:
        train_dir (str): Path to the training directory.
        labels (list): List of label names.

    Returns:
        dict: A dictionary containing the multipliers, final counts, and increase needed for each label.
    """
    # Count the number of images for each label in the training directory
    initial_counts = {
        label: len(os.listdir(os.path.join(train_dir, label))) if os.path.exists(os.path.join(train_dir, label)) else 0
        for label in labels
    }

    # Define the total target count (maximum total count after balancing)
    max_target = max(initial_counts.values())

    # Calculate the required multiplier (image_num) for each label
    balanced_multipliers = {
        label: ceil(max_target / count) if count > 0 else 1
        for label, count in initial_counts.items()
    }

    # Calculate the final balanced count for each label
    final_counts = {
        label: multiplier * count for label, count, multiplier in zip(
            initial_counts.keys(), initial_counts.values(), balanced_multipliers.values()
        )
    }

    # Calculate the increase needed to reach balance
    increase_needed = {
        label: final_count - count for label, count, final_count in zip(
            initial_counts.keys(), initial_counts.values(), final_counts.values()
        )
    }

    # In giá trị ra màn hình
    print("Initial counts:", initial_counts)
    print("Multipliers:", balanced_multipliers)
    print("Final counts:", final_counts)
    print("Increase needed:", increase_needed)
    
    return {
        "initial_counts": initial_counts,
        "multipliers": balanced_multipliers,
        "final_counts": final_counts,
        "increase_needed": increase_needed
    }
    
@tf.function
def predict_batch(model, inputs):
    """
    Hàm dự đoán hàng loạt để giảm retracing TensorFlow.
    Args:
        model (tf.keras.Model): Mô hình TensorFlow.
        inputs (np.array): Dữ liệu đầu vào.

    Returns:
        np.array: Dự đoán đầu ra của mô hình.
    """
    return model(inputs)


from skimage.filters import gabor

def extract_gabor_features(image):
    gray_image = rgb2gray(image)
    gabor_features = []
    for theta in range(4):  # Orientations: 0, 45, 90, 135 degrees
        theta = theta / 4.0 * np.pi
        for frequency in (0.1, 0.3, 0.5):
            filt_real, _ = gabor(gray_image, frequency=frequency, theta=theta)
            gabor_features.append(filt_real.mean())
            gabor_features.append(filt_real.var())
    return np.array(gabor_features)

from skimage.feature import graycomatrix, graycoprops

def extract_glcm_features(image):
    gray_image = (rgb2gray(image) * 255).astype(np.uint8)
    glcm = graycomatrix(
        gray_image,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        symmetric=True,
        normed=True
    )
    contrast = graycoprops(glcm, 'contrast').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    energy = graycoprops(glcm, 'energy').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    return np.array([contrast, correlation, energy, homogeneity])

import pywt

def extract_wavelet_features(image):
    gray_image = rgb2gray(image)
    coeffs = pywt.wavedec2(gray_image, 'db1', level=2)
    wavelet_features = []
    for coeff in coeffs:
        wavelet_features.extend([np.mean(coeff), np.var(coeff)])
    return np.array(wavelet_features)

def extract_fractal_features(image):
    gray_image = rgb2gray(image)
    size = gray_image.shape[0]
    thresholds = np.linspace(0, 1, 10)
    box_counts = []
    for thresh in thresholds:
        binary = gray_image > thresh
        box_counts.append(np.sum(binary))
    return np.array(box_counts)


from skimage import measure
from skimage.feature import canny
def extract_edge_features(image):
    """
    Trích xuất đặc trưng từ các cạnh và đường viền.
    """
    gray_image = rgb2gray(image)
    edges = canny(gray_image)
    contours = measure.find_contours(edges, 0.8)
    contour_lengths = [len(contour) for contour in contours]
    return np.array([np.mean(contour_lengths), np.std(contour_lengths)])

def extract_hog_features(image):
    """Trích xuất đặc trưng HOG."""
    gray_image = rgb2gray(image)
    return hog(gray_image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(2, 2), 
               block_norm='L2-Hys', visualize=False)

def extract_lbp_features(image, radius=1, n_points=8):
    """Trích xuất đặc trưng LBP."""
    gray_image = rgb2gray(image)
    lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26), density=True)
    return lbp_hist

def extract_color_histogram(image, bins=(8, 8, 8)):
    """Trích xuất histogram màu RGB."""
    hist_rgb = [np.histogram(image[:, :, i], bins=bins[i], range=(0, 256), density=True)[0] for i in range(3)]
    return np.concatenate(hist_rgb)

def extract_hsv_histogram(image, bins=(8, 8, 8)):
    """
    Trích xuất histogram HSV từ hình ảnh.
    Args:
        image (np.array): Hình ảnh đầu vào.
        bins (tuple): Số lượng bins cho các kênh HSV.

    Returns:
        np.array: Histogram HSV đã được trích xuất.
    """
    hsv_image = rgb2hsv(image)
    hist_hsv = [
        np.histogram(hsv_image[:, :, i], bins=bins[i], range=(0, 1), density=True)[0]
        for i in range(3)
    ]
    hist_hsv = np.nan_to_num(hist_hsv)  # Thay NaN bằng 0 để tránh lỗi
    return np.concatenate(hist_hsv)

def extract_color_correlation(image):
    """Extract color correlation features between RGB channels."""
    r_channel = image[:, :, 0].flatten()
    g_channel = image[:, :, 1].flatten()
    b_channel = image[:, :, 2].flatten()
    rg_corr = np.corrcoef(r_channel, g_channel)[0, 1]
    rb_corr = np.corrcoef(r_channel, b_channel)[0, 1]
    gb_corr = np.corrcoef(g_channel, b_channel)[0, 1]
    return np.array([rg_corr, rb_corr, gb_corr])


def extract_features_v2(image_path, label, folder_out, image_num, layers, model):
    valid_extensions = (".jpg", ".png", ".jpeg")  # Định dạng tệp hợp lệ
    if not image_path.endswith(valid_extensions):
        return  # Bỏ qua các tệp không hợp lệ
    
    # Preprocess images for different models with fixed input sizes
    try:
        # General preprocessing for most models
        img = load_img(image_path, target_size=(224, 224))
        img = img_to_array(img)
        img = preprocess_input(img)

        # VGG, ResNet, MobileNet, EfficientNet, DenseNet
        img_224 = load_img(image_path, target_size=(224, 224))
        img_224 = img_to_array(img_224)
        img_224 = preprocess_input(img_224)

        # InceptionV3
        img_299 = load_img(image_path, target_size=(299, 299))
        img_299 = img_to_array(img_299)
        img_299 = preprocess_input(img_299)

        # Preprocess for ViT (adjust to 224x224)
        img_224_vit = load_img(image_path, target_size=(224, 224))
        img_224_vit = img_to_array(img_224_vit)
        img_224_vit = preprocess_input(img_224_vit)
        img_224_vit = np.expand_dims(img_224_vit, axis=0)  # Add batch dimension

        # img_efficientnetb7 
        img_600 = load_img(image_path, target_size=(600, 600))
        img_600 = img_to_array(img_600)
        img_600 = preprocess_input(img_600)
    
    except Exception as e:
        print(f"[ERROR] Error preprocessing image {image_path}: {e}")
        return

    image_name = os.path.basename(image_path)

    # Create output directories for additional features
    feature_dirs = {
        "hog": os.path.join(folder_out, 'hog_features', label),
        "lbp": os.path.join(folder_out, 'lbp_features', label),
        "color_hist": os.path.join(folder_out, 'color_histograms_features', label),
        "hsv_hist": os.path.join(folder_out, 'hsv_histograms_features', label),
        "fc2": os.path.join(folder_out, 'fc2_features', label),
        "gabor": os.path.join(folder_out, 'gabor_features', label),
        "glcm": os.path.join(folder_out, 'glcm_features', label),
        "wavelet": os.path.join(folder_out, 'wavelet_features', label),
        "fractal": os.path.join(folder_out, 'fractal_features', label),
        "edge": os.path.join(folder_out, 'edge_features', label),
        "color_corr": os.path.join(folder_out, 'color_correlation_features', label),
        "vgg16": os.path.join(folder_out, 'vgg16_features', label),
        "vgg19": os.path.join(folder_out, 'vgg19_features', label),
        "resnet50": os.path.join(folder_out, 'resnet50_features', label),
        "resnet101": os.path.join(folder_out, 'resnet101_features', label),
        "resnet152": os.path.join(folder_out, 'resnet152_features', label),
        "inceptionv3": os.path.join(folder_out, 'inceptionv3_features', label),
        "mobilenet": os.path.join(folder_out, 'mobilenet_features', label),
        "efficientnetb0": os.path.join(folder_out, 'efficientnetb0_features', label),
        "efficientnetb7": os.path.join(folder_out, 'efficientnetb7_features', label),
        "densenet121": os.path.join(folder_out, 'densenet121_features', label),
        "densenet169": os.path.join(folder_out, 'densenet169_features', label),
        "densenet201": os.path.join(folder_out, 'densenet201_features', label),
        "vit_b16": os.path.join(folder_out, 'vit_b16_features', label), 
    }

    for dir_path in feature_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        
    # Extract layer models specifically for VGG16
    vgg16_model = model['VGG16']
    layer_models = {
        layer_name: Model(inputs=vgg16_model.input, outputs=vgg16_model.get_layer(layer_name).output)
        for layer_name in layers
    }

    # Call feature extraction functions
    hog_feature = extract_hog_features(img)
    lbp_feature = extract_lbp_features(img)
    color_hist_feature = extract_color_histogram(img)
    hsv_hist_feature = extract_hsv_histogram(img)
    gabor_feature = extract_gabor_features(img)
    glcm_feature = extract_glcm_features(img)
    wavelet_feature = extract_wavelet_features(img)
    fractal_feature = extract_fractal_features(img)
    edge_feature = extract_edge_features(img)
    color_corr_features = extract_color_correlation(img)

    # Trích rút các đặc trưng từ mô hình deep learning
    vgg16_model = model['VGG16']
    vgg19_model = model['VGG19']
    resnet50_model = model['ResNet50']
    resnet101_model = model['ResNet101']
    resnet152_model = model['ResNet152']
    inceptionv3_model = model['InceptionV3']
    mobilenet_model = model['MobileNet']
    efficientnetb0_model = model['EfficientNetB0']
    efficientnetb7_model = model['EfficientNetB7']
    densenet121_model = model['DenseNet121']
    densenet169_model = model['DenseNet169']
    densenet201_model = model['DenseNet201']
    vit_b16_model = model['ViT_B16']

    # Extract features for each model
    try:
        # Models with 224x224 input size
        vgg16_feature = model['VGG16'].predict(np.expand_dims(img_224, axis=0)).flatten()
        vgg19_feature = model['VGG19'].predict(np.expand_dims(img_224, axis=0)).flatten()
        resnet50_feature = model['ResNet50'].predict(np.expand_dims(img_224, axis=0)).flatten()
        resnet101_feature = model['ResNet101'].predict(np.expand_dims(img_224, axis=0)).flatten()
        resnet152_feature = model['ResNet152'].predict(np.expand_dims(img_224, axis=0)).flatten()
        mobilenet_feature = model['MobileNet'].predict(np.expand_dims(img_224, axis=0)).flatten()
        efficientnetb0_feature = model['EfficientNetB0'].predict(np.expand_dims(img_224, axis=0)).flatten()
        densenet121_feature = model['DenseNet121'].predict(np.expand_dims(img_224, axis=0)).flatten()
        densenet169_feature = model['DenseNet169'].predict(np.expand_dims(img_224, axis=0)).flatten()
        densenet201_feature = model['DenseNet201'].predict(np.expand_dims(img_224, axis=0)).flatten()

        efficientnetb7_feature = model['EfficientNetB7'].predict(np.expand_dims(img_600, axis=0)).flatten()
        # Models with 299x299 input size
        inceptionv3_feature = model['InceptionV3'].predict(np.expand_dims(img_299, axis=0)).flatten()

        # Models size (ViT)
        vit_b16_feature = model['ViT_B16'].predict(img_224_vit).flatten()
        print("[INFO] Features extracted successfully.")

    except Exception as e:
        print(f"[ERROR] Error extracting features: {e}")
    
    # Save individual features to respective directories
    np.save(os.path.join(feature_dirs["hog"], f'{image_name}_hog_vector.npy'), hog_feature)
    np.save(os.path.join(feature_dirs["lbp"], f'{image_name}_lbp_vector.npy'), lbp_feature)
    np.save(os.path.join(feature_dirs["color_hist"], f'{image_name}_color_hist_vector.npy'), color_hist_feature)
    np.save(os.path.join(feature_dirs["hsv_hist"], f'{image_name}_hsv_hist_vector.npy'), hsv_hist_feature)
    np.save(os.path.join(feature_dirs["gabor"], f'{image_name}_gabor_vector.npy'), gabor_feature)
    np.save(os.path.join(feature_dirs["glcm"], f'{image_name}_glcm_vector.npy'), glcm_feature)
    np.save(os.path.join(feature_dirs["wavelet"], f'{image_name}_wavelet_vector.npy'), wavelet_feature)
    np.save(os.path.join(feature_dirs["fractal"], f'{image_name}_fractal_vector.npy'), fractal_feature)
    np.save(os.path.join(feature_dirs["edge"], f'{image_name}_edge_vector.npy'), edge_feature)
    np.save(os.path.join(feature_dirs["color_corr"], f'{image_name}_color_corr.npy'), color_corr_features)
    
    # Save features to respective directories
    np.save(os.path.join(feature_dirs["vgg16"], f'{image_name}_vgg16.npy'), vgg16_feature)
    np.save(os.path.join(feature_dirs["vgg19"], f'{image_name}_vgg19.npy'), vgg19_feature)
    np.save(os.path.join(feature_dirs["resnet50"], f'{image_name}_resnet50.npy'), resnet50_feature)
    np.save(os.path.join(feature_dirs["resnet101"], f'{image_name}_resnet101.npy'), resnet101_feature)
    np.save(os.path.join(feature_dirs["resnet152"], f'{image_name}_resnet152.npy'), resnet152_feature)
    np.save(os.path.join(feature_dirs["mobilenet"], f'{image_name}_mobilenet.npy'), mobilenet_feature)
    np.save(os.path.join(feature_dirs["efficientnetb0"], f'{image_name}_efficientnetb0.npy'), efficientnetb0_feature)
    np.save(os.path.join(feature_dirs["efficientnetb7"], f'{image_name}_efficientnetb7.npy'), efficientnetb7_feature)
    np.save(os.path.join(feature_dirs["densenet121"], f'{image_name}_densenet121.npy'), densenet121_feature)
    np.save(os.path.join(feature_dirs["densenet169"], f'{image_name}_densenet169.npy'), densenet169_feature)
    np.save(os.path.join(feature_dirs["densenet201"], f'{image_name}_densenet201.npy'), densenet201_feature)
    np.save(os.path.join(feature_dirs["inceptionv3"], f'{image_name}_inceptionv3.npy'), inceptionv3_feature)
    np.save(os.path.join(feature_dirs["vit_b16"], f'{image_name}_vit_b16.npy'), vit_b16_feature)

    # Extract features for each layer
    for layer_name, intermediate_layer_model in layer_models.items():
        feature_maps = intermediate_layer_model.predict(np.expand_dims(img, axis=0))
        _, height, width, num_filters = feature_maps.shape

        # Create directories
        layer_feature_maps_out = os.path.join(folder_out, 'blocknet_feature_map', label, layer_name)
        layer_vgg16_fc_2_features_out = os.path.join(folder_out, 'blocknet_features', label, layer_name)
        combine_features_out = os.path.join(folder_out, 'blocknet_handcrafts_features', label, layer_name)

        os.makedirs(layer_feature_maps_out, exist_ok=True)
        os.makedirs(layer_vgg16_fc_2_features_out, exist_ok=True)
        os.makedirs(combine_features_out, exist_ok=True)

        # Debugging feature map dimensions
        print(f"[DEBUG] Feature maps shape: {feature_maps.shape}")

        # Kiểm tra nếu feature_maps có đúng số chiều (batch, height, width, channels)
        if len(feature_maps.shape) != 4:
            print(f"[ERROR] Unexpected feature map shape: {feature_maps.shape}")
            return

        # Duyệt qua số lượng kênh hợp lệ
        entropies = []
        try:
            for i in range(feature_maps.shape[-1]):
                # Truy cập và resize feature map tại kênh i
                resized_map = resize(feature_maps[0, :, :, i], (224, 224))  # Truy cập đúng chiều
                entropy_value = check_entropy(resized_map)  # Tính entropy
                entropies.append(entropy_value)
        except IndexError as e:
            print(f"[ERROR] Indexing error: {e}")
            return
        
        min_entropy, max_entropy = determine_optimal_entropy_range(entropies)

        # Lọc các feature maps trong khoảng tối ưu
        # feature_map_with_entropy = [(i, feature_maps[:, :, i], entropy) for i, entropy in enumerate(entropies) if min_entropy <= entropy <= max_entropy]
        feature_map_with_entropy = [
            (i, feature_maps[0, :, :, i], entropy)
            for i, entropy in enumerate(entropies)
            if min_entropy <= entropy <= max_entropy
        ]
        # Process and save features
        for i in range(min(image_num, len(feature_map_with_entropy))):
            feature_idx, feature_map, entropy = feature_map_with_entropy[i]
            resized_feature_map = resize(feature_map, (224, 224))

            # Đảm bảo resized_feature_map là mảng 2D
            if resized_feature_map.ndim > 2:
                resized_feature_map = np.mean(resized_feature_map, axis=-1)  # Chuyển thành grayscale bằng cách lấy trung bình

            # Save the feature_map image
            feature_map_image_path = os.path.join(layer_feature_maps_out, f'{image_name}_feature_map_{feature_idx}.jpg')
            feature_map_image = Image.fromarray((resized_feature_map * 255).astype(np.uint8), 'L')
            feature_map_image.save(feature_map_image_path)


            # Process feature map for vgg16_fc_2
            resized_feature_map_rgb = np.stack([resized_feature_map] * 3, axis=-1)  # Convert grayscale to RGB
            feature_map_batch = np.expand_dims(resized_feature_map_rgb, axis=0)  # Add batch dimension
            feature_map_batch = preprocess_input(feature_map_batch)

            vgg16_fc_2_feature = vgg16_model.predict(feature_map_batch).flatten()

            # Save the fc-2 feature
            vgg16_fc_2_feature_path = os.path.join(layer_vgg16_fc_2_features_out, f'{image_name}_feature_map_vector_{feature_idx}.npy')
            np.save(vgg16_fc_2_feature_path, vgg16_fc_2_feature)


            # Representative features for each category
            color_corr_features = extract_color_correlation(img)
            # Selected texture feature (GLCM)
            texture_feature = extract_glcm_features(img)

            # Selected shape feature (HOG)
            shape_feature = extract_hog_features(img)

            # Optionally include the deep feature (VGG16 FC2)
            deep_feature = vgg16_fc_2_feature

            # Combine all selected features into a single feature vector
            combined_feature = np.concatenate([
                    color_corr_features, texture_feature, shape_feature, deep_feature
                ])

            # Save the combined feature
            combined_feature_path = os.path.join(combine_features_out, f'{image_name}_combine_handcrafts_and_feature_map_vector_{i}.npy')
            np.save(combined_feature_path, combined_feature)

            
def write_layer_statistics_to_file(layer_stats, output_file):
    """
    Ghi thống kê nhãn theo từng layer vào tệp txt.
    Args:
        layer_stats (dict): Thống kê theo từng layer.
        output_file (str): Đường dẫn đến tệp txt để ghi thống kê.
    """
    with open(output_file, "w") as f:
        f.write("Layer-wise Label Statistics:\n")
        for dataset, dataset_stats in layer_stats.items():
            f.write(f"\nStatistics for {dataset} dataset:\n")
            for layer_name, layer_data in dataset_stats.items():
                f.write(f"  Layer: {layer_name}\n")
                for label, count in layer_data.items():
                    f.write(f"    {label}: {count}\n")
    print(f"[INFO] Layer-wise statistics saved to {output_file}")


def main():
    """
    Main function to balance datasets, extract features, and generate statistics.
    """
    layers = ['block1_conv1', 'block1_conv2', 
              'block2_conv1', 'block2_conv2', 
              'block3_conv1', 'block3_conv2', 'block3_conv3',
              'block4_conv1', 'block4_conv2', 'block4_conv3', 
              'block5_conv1', 'block5_conv2', 'block5_conv3']

    # Load the SOTA models
    model = load_SOTA_models()
    
    # Đường dẫn đến thư mục chứa dữ liệu
    home_dir = os.getcwd()
    data_dir = os.path.join(home_dir, 'data2')
    
    data_input_dir = os.path.join(data_dir, 'data2_HAM10000_SPLIT')
    train_input_dir = os.path.join(data_input_dir, 'train')
    val_input_dir = os.path.join(data_input_dir, 'val')
    test_input_dir = os.path.join(data_input_dir, 'test')
    
    # Các lớp nhãn trong dữ liệu
    labels = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

    # Tạo các thư mục đầu ra
    feature_output_dir = os.path.join(data_dir, 'data2_SOTA_and_handcrafts_and_BlookNet_optimal_entropy_features')
    os.makedirs(feature_output_dir, exist_ok=True)
    
    train_output_dir = os.path.join(feature_output_dir, 'train')
    os.makedirs(train_output_dir, exist_ok=True)
    
    val_output_dir = os.path.join(feature_output_dir, 'val')
    os.makedirs(val_output_dir, exist_ok=True)
    
    test_output_dir = os.path.join(feature_output_dir, 'test')
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Dictionary to store layer-wise statistics for all datasets
    layer_stats = {"train": {}, "val": {}, "test": {}}

    # Function to initialize layer statistics
    def initialize_layer_stats():
        return {layer: {label: 0 for label in labels} for layer in layers}

    # Xử lý tập train
    print("Processing train dataset...")
    layer_stats["train"] = initialize_layer_stats()
    train_balanced_details = balance_dataset(train_input_dir, labels)

    for label, multiplier in train_balanced_details["multipliers"].items():
        imgnum = multiplier
        label_dir = os.path.join(train_input_dir, label)
        if os.path.exists(label_dir):
            for filename in os.listdir(label_dir):
                image_path = os.path.join(label_dir, filename)
                extract_features_v2(
                    image_path, label,
                    train_output_dir,
                    imgnum, layers, model
                )
                for layer in layers:
                    layer_stats["train"][layer][label] += 1  # Update statistics
        print(f"Train Dataset {label} completed.")


    # Xử lý tập val
    print("Processing val dataset...")
    layer_stats["val"] = initialize_layer_stats()
    for label in labels:
        imgnum = 1
        label_dir = os.path.join(val_input_dir, label)
        if os.path.exists(label_dir):
            for filename in os.listdir(label_dir):
                image_path = os.path.join(label_dir, filename)
                extract_features_v2(
                    image_path, label,
                    val_output_dir,
                    imgnum,  # Chỉ lấy 1 ảnh
                    layers=layers, model=model
                )
                for layer in layers:
                    layer_stats["val"][layer][label] += 1  # Update statistics
        print(f"Val Dataset {label} completed.")

    # Xử lý tập test
    print("Processing test dataset...")
    layer_stats["test"] = initialize_layer_stats()
    for label in labels:
        imgnum = 1
        label_dir = os.path.join(test_input_dir, label)
        if os.path.exists(label_dir):
            for filename in os.listdir(label_dir):
                image_path = os.path.join(label_dir, filename)
                extract_features_v2(
                    image_path, label,
                    test_output_dir,
                    imgnum,  # Chỉ lấy 1 ảnh
                    layers=layers, model=model
                )
                for layer in layers:
                    layer_stats["test"][layer][label] += 1  # Update statistics
        print(f"Test Dataset {label} completed.")

    # Ghi tất cả thống kê theo từng layer vào một tệp chung
    stats_output_file = os.path.join(feature_output_dir, "train_layer_wise_label_statistics.txt")
    write_layer_statistics_to_file(layer_stats, stats_output_file)
        
if __name__ == "__main__":
    main()
