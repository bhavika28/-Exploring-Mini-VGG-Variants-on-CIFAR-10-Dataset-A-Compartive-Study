# -Exploring-Mini-VGG-Variants-on-CIFAR-10-Dataset-A-Compartive-Study



### Project Title:
**"Exploring Mini-VGG Variants on CIFAR-10 Dataset: A Comparative Study of Model Performance and Optimization Techniques"**

### Steps for the Project:

#### Step 1: Load CIFAR-10 Dataset
- Import the CIFAR-10 dataset using PyTorch or TensorFlow.
- Split the dataset into training, validation, and test sets.
- Normalize the pixel values between 0 and 1 for optimal performance.

#### Step 2: Preprocess the Data
- Standardize the input data by normalizing pixel values to a range of [0, 1].
- Convert labels to categorical for multi-class classification.

---

### Mini-VGG Model: Implementation and Performance Evaluation

#### Step 3: Implement the Mini-VGG Model
- Design the mini-VGG model with the following architecture:
  1. Conv3 – 64 filters
  2. Conv3 – 64 filters
  3. Maxpool – 2x2
  4. Conv3 – 128 filters
  5. Conv3 – 128 filters
  6. Maxpool – 2x2
  7. Conv3 – 256 filters
  8. Conv3 – 256 filters
  9. Maxpool – 2x2
  10. Fully Connected Layer – 512 units
  
- Use **ReLU activation** for all convolutional and fully connected layers, except the last one.
- Add a reshape layer before the fully connected layer to prepare the data for dense layers.

#### Step 4: Model Training & Validation
- Use **CrossEntropyLoss** (PyTorch) or **softmax cross entropy** (TensorFlow) for the loss function.
- Train the model on the training set and tune hyperparameters using the validation set.
- Evaluate the model on the test set after training and report the test accuracy.

#### Step 5: Performance Evaluation and Visualization
- Plot **Training Loss vs. Validation Loss**.
- Plot **Training Accuracy vs. Validation Accuracy**.
- Calculate and report **Test Accuracy**.

---

### Variant 1: Replace ReLU with SELU and Swish Activation Functions

#### Step 6: Modify Activation Functions
- Replace the **ReLU** activation function with **SELU** and **Swish**.
- Train and evaluate the modified models using the same dataset.

#### Step 7: Performance Comparison
- Compare the performance of these models with the original Mini-VGG model.
- Plot the performance metrics (training/validation loss and accuracy).
- Discuss whether changing the activation function leads to improved performance.

---

### Variant 2: Removing MaxPooling and Using Stride=2

#### Step 8: Replace MaxPooling Layers with Stride=2
- Remove all **MaxPooling** layers from the Mini-VGG model.
- In the convolutional layers preceding each MaxPooling, set **stride=2** to achieve similar size reduction.

#### Step 9: Train the Modified Model
- Train the modified model and evaluate it using the same dataset and metrics.

#### Step 10: Performance Comparison
- Compare the performance of this variant with the original Mini-VGG model.
- Analyze whether this modification improves accuracy or reduces overfitting.

---

### Variant 3: Adding Dropout Layers

#### Step 11: Introduce Dropout Layers
- Add **Dropout** layers in different positions of the network.
- Implement two variants:
  1. Add dropout after each MaxPooling layer.
  2. Add dropout before the fully connected layer.

#### Step 12: Train the Dropout Models
- Train the models with dropout layers and evaluate the performance.

#### Step 13: Performance Evaluation
- Compare the two Dropout variants with the original Mini-VGG.
- Discuss the impact of Dropout on model regularization and performance.

---

### Variant 4: All-Convolutional Architecture

#### Step 14: Remove Fully Connected Layers
- Remove layers 9 and 10 (fully connected layers).
- Add two 1x1 convolutional layers:
  1. Conv(1,1) – 128 filters
  2. Conv(1,1) – 10 filters
- Add **GlobalAveragePooling2D** to merge the feature maps.

#### Step 15: Train the All-Convolutional Model
- Train the modified all-convolutional model and report the performance.

#### Step 16: Compare the All-Convolutional Model
- Compare this all-convolutional model with the original Mini-VGG.
- Analyze how the removal of fully connected layers affects performance, model size, and training time.

---

### Summary Report:
1. **Results Summary**:
   - Summarize the results of each experiment, including the Mini-VGG and its four variants.
   
2. **Classification Performance**:
   - Discuss the classification accuracy for each model variant on the test set.

3. **Model Size**:
   - Report the number of parameters for each variant.
   - Highlight how removing or adding layers affected the model size.

4. **Computation Time**:
   - Compare the training time for each model variant and analyze the time efficiency.
