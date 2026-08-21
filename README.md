# 3MP Multimodal Diagnostic Platform: Deep Learning Pipeline

## Overview

This repository contains the source code, normalized sensor measurements, processed spatial tensors, and cross-validation results for the multimodal convolutional neural network (CNN) described in the manuscript *An Integrated Multimodal Magnetoelastic Microneedle Patch for Rapid Skin Cancer Screening*.

The computational framework integrates spatially resolved biomechanical and biochemical measurements acquired using the 3MP platform to classify four tissue phenotypes: normal tissue, scar, inflammation, and tumor.

## Computational Workflow

The analysis consists of three sequential stages:

**1. Experimental sensor matrices**

Normalized biomechanical and biochemical measurements are provided as 3 x 3 sensor matrices in `Normalized_Sensor_Tensors/`.

**2. Spatial preprocessing**

Before CNN training, each normalized 3 x 3 sensor matrix was converted into a 96 x 96 spatial tensor using bicubic interpolation. The resulting processed biomechanical and biochemical tensors are provided in `Skin_Cancer_Dataset/`.

The CNN training notebook therefore uses the supplied 96 x 96 tensors directly and does not repeat the bicubic interpolation step.

**3. CNN classification**

`3MP_Multimodal_Pipeline.ipynb` performs five-fold stratified cross-validation using biomechanical-only, biochemical-only, and multimodal inputs. Out-of-fold predictions are generated for every sample, allowing direct comparison among the three models.

The workflow can therefore be summarized as:

`3 x 3 normalized sensor measurements -> bicubic spatial interpolation -> 96 x 96 tensors -> five-fold CNN cross-validation -> out-of-fold predictions -> ROC/confusion-matrix analysis`

## Repository Structure

```text
3MP-Multimodal-Diagnostic-Platform/
|
|-- Normalized_Sensor_Tensors/
|   |-- Biochemistry data.xlsx
|   `-- Biomechanical data.xlsx
|
|-- Skin_Cancer_Dataset/
|   |-- Normal/
|   |-- Scar/
|   |-- Inflamed/
|   `-- Tumor/
|
|-- Publication_Results/
|   |-- Ablation_ROC_Probabilities.csv
|   |-- Bio_tSNE_Coordinates.csv
|   |-- Mech_tSNE_Coordinates.csv
|   `-- Fused_tSNE_Coordinates.csv
|
|-- Source_Code/
|   |-- 3MP_Multimodal_Pipeline.ipynb
|   `-- Figure_Generator.ipynb
|
|-- requirement.txt
|-- LICENSE
`-- README.md
```

### `Normalized_Sensor_Tensors/`

Contains the normalized experimental 3 x 3 biomechanical and biochemical sensor measurements prior to spatial interpolation.

### `Skin_Cancer_Dataset/`

Contains the paired 96 x 96 biomechanical and biochemical tensors used directly for CNN training and evaluation.

The dataset contains 384 samples distributed among four tissue classes:

* Normal: 74
* Scar: 75
* Inflamed: 72
* Tumor: 163

For each sample, the biomechanical and biochemical tensors are spatially registered and combined as separate input channels for multimodal analysis.

### `Publication_Results/`

Contains the reference out-of-fold prediction probabilities and derived data used to generate the manuscript figures.

These files correspond to the computational results used for the reported publication analyses and are retained separately from newly reproduced training outputs.

### `Source_Code/3MP_Multimodal_Pipeline.ipynb`

Primary machine-learning notebook. It:

* loads the supplied 96 x 96 biomechanical and biochemical tensors;
* verifies sample pairing, tensor dimensions, and class counts;
* constructs the attention-based CNN;
* performs five-fold stratified cross-validation;
* evaluates biomechanical-only, biochemical-only, and multimodal models;
* performs test-time augmented inference; and
* exports out-of-fold class probabilities.

Independent reruns are written to `Reproduced_Results/` so that the fixed reference data in `Publication_Results/` are not overwritten.

### `Source_Code/Figure_Generator.ipynb`

Reads the fixed source data in `Publication_Results/` and reproduces the ROC curves, confusion matrices, and t-SNE plots as vector graphics.

Newly generated figures are written to `Generated_Figures/`.

## Cross-Validation and Reproducibility

Five-fold stratified cross-validation is used with a fixed random seed (`SEED = 42`). Each sample is used as a held-out test sample exactly once, and the resulting predictions are stored as out-of-fold probabilities.

The same fold partition is used for the biomechanical-only, biochemical-only, and multimodal models.

Random seeds are specified for Python, NumPy, TensorFlow, data shuffling, data augmentation, and dropout. Deterministic TensorFlow operations are enabled when supported.

Because numerical implementations can differ among TensorFlow, CUDA, cuDNN, GPU, and operating-system environments, independently retrained models may not reproduce the exact stochastic realization of the original training run. The fixed results in `Publication_Results/` correspond to the source data used for the manuscript figures.

The current pipeline was successfully re-executed in Google Colab using TensorFlow 2.20.0.

## Running the CNN Analysis

Clone the repository:

```bash
git clone https://github.com/GanghaoLiang/3MP-Multimodal-Diagnostic-Platform.git
cd 3MP-Multimodal-Diagnostic-Platform
```

Install the required packages:

```bash
pip install -r requirement.txt
```

Open:

```text
Source_Code/3MP_Multimodal_Pipeline.ipynb
```

and execute Cells 1-3 sequentially.

The data-loading stage should report:

```text
Loaded  74 samples from Normal.
Loaded  75 samples from Scar.
Loaded  72 samples from Inflamed.
Loaded 163 samples from Tumor.

X_data shape:      (384, 96, 96, 2)
Y_labels shape:    (384,)
Class counts:      [ 74  75  72 163]

Dataset integrity check PASSED.
```

The training notebook creates `Reproduced_Results/` and exports the independently reproduced out-of-fold probabilities without overwriting the manuscript reference results.

## Reproducing the Publication Figures

Open:

```text
Source_Code/Figure_Generator.ipynb
```

and execute the notebook.

The notebook reads the fixed source data from:

```text
Publication_Results/
```

and writes newly generated figures to:

```text
Generated_Figures/
```

The generated outputs include three confusion matrices, four class-specific ROC plots, and three t-SNE feature-space plots.

A successful run should reproduce the reference classification accuracies:

```text
Biochemical-only:   60.94%
Biomechanical-only: 71.61%
Multimodal fusion:  89.32%
```

## Notes

The 96 x 96 tensors in `Skin_Cancer_Dataset/` are the direct inputs used for CNN training. They were generated from the normalized 3 x 3 experimental sensor measurements by bicubic spatial interpolation before CNN training.

The `Publication_Results/` directory should be treated as fixed source data corresponding to the manuscript figures. Independent model reruns should be compared against, rather than overwrite, these reference results.

Newly generated model outputs are stored in:

```text
Reproduced_Results/
```

and newly generated figures are stored in:

```text
Generated_Figures/
```

This separation preserves the original source data used for manuscript figure generation.

## License

Please refer to the `LICENSE` file for licensing information.
