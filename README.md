# Project Overview
This project uses YOLOv5 to detect whether people are wearing face masks correctly. It classifies images into three categories: with mask, without mask, and mask worn incorrectly. The model was trained on a custom dataset with annotations converted from Pascal VOC to YOLO format. It includes tools for evaluation and visualization, highlighting false positives and false negatives to better understand model performance.
*[Presetation]([https://www.markdownguide.org](https://www.canva.com/design/DAGoXSkAYSo/XeGl1PJJznUXU5tzxXJykg/edit?utm_content=DAGoXSkAYSo&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton))*.

## Documentation

### Python Libraries:
#### ultralytics:
- Purpose: The core library for YOLO model operations, including loading pre-trained weights, training custom models, performing inference, and evaluating results.
- Installation: !pip install ultralytics

#### google.colab.drive:
- Purpose: Enables mounting Google Drive in Colab, essential for accessing datasets and saving trained models.
- Availability: Built-in to Google Colab.

#### os:
- Purpose: Python's standard library for interacting with the operating system, used for directory creation.
- Availability: Built-in to Python.

#### shutil:
- Purpose: Python's high-level file operations module, used for copying directories (e.g., saving training results to Drive).
- Availability: Built-in to Python.

#### PIL (Pillow):
- Purpose: Python Imaging Library, used for image manipulation and displaying prediction results.
- Installation: Usually a dependency of ultralytics; otherwise, !pip install Pillow.

#### File: yolov5n.pt
Purpose: Serves as a base for transfer learning, automatically downloaded by ultralytics.
Dataset Archive:

### YOLO Dataset Configuration:
#### File: mask_detection.yaml
- Purpose: Defines the dataset structure, including paths to images, number of classes, and class names, for YOLO training.
Location: Should be accessible during training (e.g., in the same directory as the notebook or explicitly specified).

### Trained Model Weights:
#### File: best.pt
- Purpose: The best performing weights saved after training, used for custom predictions and evaluation.


### Functions
```
__init__(self)
```
- Basic initializer for a class (likely used for setting defaults).

```
__init__(self, model, target_class_name)
```
- Initializes a class with a YOLO model and a target class name for analysis.

```
analyze_face_contrast(self, image, face_region)
```
- Analyzes contrast within a detected face region—possibly to detect improperly worn masks.

```
augment_for_contrast_detection(self, image)
```
- Applies image augmentation specifically designed to enhance contrast detection.

```
convert_bbox(size, box)
```
- Converts bounding box coordinates from one format (likely VOC) to YOLO format using image size.

```
convert_annotations_to_yolo(xml_folder, txt_folder)
```
- Parses XML annotation files (Pascal VOC format) and saves them as YOLO .txt files.

```
move_files(file_list, src_img, src_lbl, dst_img, dst_lbl)
```
- Moves a selected list of image and label files from source folders to destination folders.

```
copy_labels_for_images(img_dir, label_src, label_dst)
```
- Copies label files that match the images in a directory from a source to a destination.

```
load_yolo_labels(txt_path, img_shape)
```
- Loads YOLO-format label files and rescales bounding boxes to pixel coordinates based on image shape.

```
calculate_iou(box1, box2)
```
- Calculates Intersection over Union (IoU) between two bounding boxes. Useful for evaluation and filtering.

```
visualize_predictions_with_stats(model, val_image_folder, labels_folder, classes, num_images, iou_thresh, conf_thresh)
```
- Runs inference on validation images and visualizes predicted bounding boxes with stats like IoU and confidence.

```
compute_confusion_stats(model)
```
- Performs inference and collects prediction statistics per class to compute confusion matrix values.

```
plot_confusion_stats(df_stats, class_names)
```
- Plots confusion matrix or class-wise statistics using seaborn from a given stats DataFrame.

```
get_images_with_class(model, target_class_name, image_folder, num_images, image_size)
```
- Finds and returns images from a folder where the model detected a specified class.

```
get_images_with_class_from_list(model, target_class_name, image_list, image_folder, image_size)
```
- Similar to get_images_with_class, but uses a predefined list of image names.

```
detailed_lime_explanation(model, target_class_name, image_paths, image_size)
```
- Generates LIME visualizations to explain why the model classified an image as a specific class.

```
compare_predictions_and_explanations(model, target_class_name, image_paths)
```
- Compares the YOLO predictions with corresponding LIME explanations to highlight model reasoning.

```
process_yolo_dataset_with_contrast_universal(dataset_dir, output_dir, analyze_existing)
```
- Processes the YOLO dataset by applying a contrast transformation (or other universal preprocessing) and saves the results.

```
copy_folder_contents(src_dir, dst_dir)
```
- Copies the contents of one folder to another, preserving structure—used to manage datasets or output.
  
```
predict(self, images)
```
- Performs batch prediction using the model on given images.

```
process_yolo_dataset_with_contrast_universal(dataset_dir, output_dir, analyze_existing=True)
```
- Processes a YOLO dataset with contrast-based logic, possibly for data cleaning or augmentation.

### Trained models:
```
model_custom
```
- possible data leakage

```
model_custom2
```
- model with default parametrs & normal data (best model)

```
model_custom3
```
- model with default parametrs & data with augmentation

```
model_custom4
```
- model with adjusted parametrs & normal data

### Visualization Tools
- matplotlib and seaborn: For plotting results
- cv2 (OpenCV): For image preprocessing
- PIL: For image manipulation
- slic and mark_boundaries: For LIME-based explanations

metrics = model_custom.val(): 
-Evaluates the performance of the model_custom4 on the validation dataset defined in mask_detection.yaml. This function calculates various metrics such as precision, recall, and mAP (mean Average Precision).
- map: Mean Average Precision across all IoU thresholds (from 0.5 to 0.95 with a step of 0.05).
- map50: Mean Average Precision at an IoU threshold of 0.50. This is a common metric for object detection.
- map75: Mean Average Precision at an IoU threshold of 0.75. This is a stricter metric.
- maps: Mean Average Precision (mAP) for each individual class.
