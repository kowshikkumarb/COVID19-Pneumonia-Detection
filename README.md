# COVID19-Pneumonia-Detection

Here's a revised summary tailored to a Python code that preprocesses images using local histogram equalization and builds a TensorFlow model to detect and distinguish between COVID-19, pneumonia, and normal cases:

---

### Summary

This Python code aims to classify chest X-ray images into three categories: **COVID-19**, **Pneumonia**, and **Normal**. The workflow includes image preprocessing and building a convolutional neural network (CNN) using TensorFlow.

1. **Image Preprocessing:**
   - **Local Histogram Equalization:** Chest X-ray images are preprocessed using local histogram equalization, specifically contrast-limited adaptive histogram equalization (CLAHE). This technique enhances the contrast of the X-ray images by redistributing the brightness of localized areas, making critical features like lung patterns more distinguishable for the model.
   - The `skimage.exposure.equalize_adapthist` function from the `skimage` library is used to perform this operation.
   - Post-equalization, the images are resized and normalized to a standard range (e.g., 0 to 1) to prepare them for model training.

2. **Model Building with TensorFlow:**
   - A **Convolutional Neural Network (CNN)** is designed to extract important features from the X-ray images.
   - The architecture includes:
     - **Convolutional layers** (`Conv2D`) to detect features such as edges, textures, and lung patterns.
     - **Pooling layers** (`MaxPooling2D`) to reduce the spatial dimensions while retaining the most important information.
     - **Fully connected layers** (`Dense`) to map the learned features to the output categories: COVID-19, Pneumonia, and Normal.
   - The model uses `softmax` activation in the output layer for multi-class classification.
   - The model is compiled using `categorical_crossentropy` as the loss function and the `Adam` optimizer, along with accuracy as the evaluation metric.

3. **Model Training:**
   - The dataset is split into training and validation sets.
   - The model is trained using the training data, while its performance is monitored on the validation set. The `fit` function is used for training with batch size and epochs defined to optimize the model.
   - Data augmentation (e.g., rotations, shifts) might be applied to improve the generalization ability of the model and prevent overfitting.

4. **Model Evaluation:**
   - After training, the model is evaluated on a test set containing unseen X-ray images.
   - Performance metrics such as accuracy, precision, recall, and F1-score are calculated to measure the model's effectiveness in distinguishing between COVID-19, Pneumonia, and Normal cases.

5. **Prediction:**
   - The trained model can then be used to classify new chest X-ray images, outputting probabilities for each class (COVID-19, Pneumonia, Normal).

This approach provides a robust method for detecting and distinguishing between COVID-19, Pneumonia, and Normal chest conditions from X-ray images, offering a potential tool for automated diagnosis in clinical settings.
