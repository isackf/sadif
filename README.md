# README.md

## Project Title

Seismic Attribute-Driven Interpretability Framework (SADIF)

## Project Overview

This project provides a complete toolchain for training, analyzing, and visualizing image segmentation models (U-Net architecture). Key features include:

* Loading and preprocessing datasets (images and annotations)
* Building and training U-Net models
* Visualizing models with Grad-CAM, confidence maps, feature maps, etc.
* Analyzing model sensitivity to occlusion or semantic masking

This project is especially suitable for researchers working on model interpretability or writing academic papers.

## Requirements

* Python 3.8+
* TensorFlow 2.x
* NumPy, Matplotlib, PIL, scikit-learn

## Project Structure

```bash
code/
├── analyser.py             # Grad-CAM and confidence map analysis tools
├── analyser_v2.py          # Advanced analysis module (multi-class support, toggle display)
├── data_loader.py          # Load images, convert labels, build tf.data.Dataset
├── load_image.py           # Load a single image and its label mask
├── model_builder.py        # Define U-Net model architecture
├── patch.py                # Patch-based occlusion analysis (spatial sensitivity)
├── patch_label.py          # Semantic class masking analysis (label-level sensitivity)
├── models_16/              # Trained model storage
├── results_tables_perturbation/ # Occlusion test result outputs
└── 1_base_64/              # Experimental modules and examples
```

## Module Descriptions

### `data_loader.py`

* `split_data`: Load and split data into train/val/test sets
* `convert_label_to_1ch`: Convert RGB labels to single-channel class masks
* `load_tf_dataset`: Wrap data into TensorFlow Datasets

### `load_image.py`

* `get_image_and_label`: Load one image and its corresponding label mask

### `model_builder.py`

* `build_unet`: Construct a customizable U-Net model

### `analyser.py` & `analyser_v2.py`

* `make_gradcam_heatmap`: Generate Grad-CAM heatmaps
* `get_confidence_map`: Compute per-pixel confidence maps
* `visualize_*`: Plotting functions (heatmaps, confidence, activation maps)

### `patch.py`

* `generate_images_with_filler`: Slide square occlusions across an image
* `evaluate_predictions`: Evaluate impact on predictions (accuracy, IoU)

### `patch_label.py`

* `generate_images_by_class`: Mask entire regions of a given class
* `evaluate_predictions`: Measure impact of semantic class masking

## Workflow

1. Use `data_loader.py` to load and split the dataset
2. Build a model with `model_builder.py` and train it externally
3. Use `load_image.py` to retrieve a test image and label
4. Run visualization with `analyser.py` or `analyser_v2.py`
5. Perform sensitivity tests using `patch.py` or `patch_label.py`

## Use Cases

* Model interpretability (XAI)
* Occlusion perturbation studies
* Visual analysis of attention regions
* Academic paper figures and quantitative metrics
