# VR Assignment 

# Objectives:
- To develop a computer vision solution to classify and segment face masks in images.
- The project involves using handcrafted features with machine learning classifiers and deep
learning techniques to perform classification and segmentation.


# Dataset:
- A labeled dataset containing images of people with and without face masks: https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset

- A Masked Face Segmentation Dataset with ground truth face masks can be accessed
here: https://github.com/sadjadrz/MFSD


# Methodology:

- **Load the dataset:**
    
```
def load_images():
    image_paths = []
    
    for label, category in enumerate(categories):
        folder_path = os.path.join(dataset_path, category)
        image_paths.extend([(os.path.join(folder_path, file), label) for file in os.listdir(folder_path)])

    X, y = [], []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_image, image_paths), total=len(image_paths), desc="Loading images"))

    # Filter out None values (failed reads)
    results = [res for res in results if res is not None]
    X, y = zip(*results)

    return np.array(X), np.array(y)
```

## Classification
  ### Using handcrafted features
- **Handcrafted features used:**
    1. HOG(Histogram of Oriented Gradient)
    2. LBP(Local Binary Patterns)
    3. Canny Edge Detectors
    4. SIFT(Scale Invariant Feature Transform)
    
**1. HOG(Histogram of Oriented Gradient):**
  
  - Extracted the HOG features of the images and stored them along with the label of their corresponding images.
  - SVM and MLP is used on histogram to classify the image. 
    ```       
    hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, block_norm='L2-Hys')
    ```

    ![image](https://github.com/user-attachments/assets/af4283de-2a41-44e9-8cf7-1248d063abac)
  - **Accuracy**
    ![image](https://github.com/user-attachments/assets/cd0b1d9f-074e-486d-ac8b-c9750140219c)

  
**2. LBP(Local Binary Patterns)**
    
  - Extracted Local Binary Pattern (LBP) features from a grayscale image and computed a histogram representation of these features.
  - The histogram summarizes the frequency of different LBP patterns in the image.
  - Again SVM and MLP is used on histogram to classify the image. 

  ```
  lbp = local_binary_pattern(gray, 8, 1, method="uniform")
  lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
  ```
  ![image](https://github.com/user-attachments/assets/c6643742-667e-44cc-bd25-59134266fc82)
  - **Accuracy**
  ![image](https://github.com/user-attachments/assets/e8ffc578-5d92-4e6b-9466-3c6e5e1f12bb)

**3. Canny Edge Detectors**
  - Used Canny edge detector to detect edges and formed histogram based on intensity level.
  - It's accuracy was very low (close to 50%) which shows it failed miserably.
  - Again SVM and MLP is used on histogram to classify the image. 

  ```
  edges = cv2.Canny(gray, 100, 200)
  np.histogram(edges.ravel(), bins=16, range=(0, 256), density=True)[0]
  ```
  ![image](https://github.com/user-attachments/assets/280a6b75-315f-40f4-98eb-1a92fba3d9ef)
  - **Accuracy**
  ![image](https://github.com/user-attachments/assets/5281bcdf-6908-45c3-ae8c-8ba705031665)
  ![image](https://github.com/user-attachments/assets/7593bb95-1ba6-46a5-a95e-d9a0bca6bd9c)

**4. SIFT(Scale Invariant Feature Transform)**
  - Keypoints and descriptor of each image was extracted.
  - The descriptors of all images were stacked together to train the K-means clustering model.
  - Then a Bag of visual Words (BoW) vector was formed by predicting the label for each keypoint in an image.
  - Again SVM and MLP is used on BoW vectors to classify the image. 
  ```
  sift = cv2.SIFT_create()
  kp, des = sift.detectAndCompute(gray, None)
  if des is None:
      return np.zeros((128,))  
  if kmeans is not None:
      labels = kmeans.predict(des)
      bow_vector = np.histogram(labels, bins=np.arange(0, kmeans.n_clusters+1), density=True)[0]
      return bow_vector
  ```
  ![image](https://github.com/user-attachments/assets/04edb065-8020-4d83-b775-455c91ee9fc5)

  - **Accuracy**
  ![image](https://github.com/user-attachments/assets/44bec42a-20c3-4888-ba6d-c36d41b851e6)


### Using CNN
  - Structure of CNN:
  ```
  def build_cnn():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        # Flatten to 1D vector
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Reduce overfitting
        Dense(2, activation='softmax')  # 2 output classes (with_mask, without_mask)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    return model
  ```
  ![image](https://github.com/user-attachments/assets/5202e325-5315-4e73-a8c8-8fdb3b5ee2c8)
  - Total params: 3,304,898 (12.61 MB)\
  Trainable params: 3,304,898 (12.61 MB)\
  Non-trainable params: 0 (0.00 B)

  - **Accuracy:**\
  ![image](https://github.com/user-attachments/assets/cccb5f9d-6a60-4d7a-b3fb-a3fb16bddd93)

  - **Loss:**\
  ![image](https://github.com/user-attachments/assets/c9b7da6f-05d9-4655-a353-5c49fe860b2c)
#### Hyperparameters and Experiments:
- **Batch size:** 32
- **Epochs:** 10,20,50,100
- **Learning rate:** 0.001,0.01
- **Optimizer:** Adam
- **Activation function:** ReLU,LeakyReLU
- **Dropout rate:** 0.5,0.4,0.3
- **Number of filters:** 32, 64, 128
- **Kernel size:** 3, 3, 3
- **Pooling size:** 2, 2, 2
- **Number of units in dense layer:** 128,256
- **Output activation function:** softmax
- **Metrics:** accuracy, sparse categorical crossentropy
- **Loss function:** sparse categorical crossentropy
- **Model architecture:** CNN with 3 convolutional layers, 2 dense layers, and dropout

| Model Variation | Optimizer | Activation | Accuracy   | Epochs |
| --------------- | --------- | ---------- | ---------- | ------ |
| CNN V1          | Adam      | ReLU       | 96.34%     | 10     |
| CNN V1          | SGD       | ReLU       | *96.70%*   | 10     |
| CNN V1          | Adam      | Tanh       | 88.64%     | 10     |
| CNN V1          | SGD       | Tanh       | 95.73%     | 10     |
|                 |           |            |            |        |
| CNN V2          | Adam      | ReLU       | 96.95%     | 10     |
| CNN V2          | SGD       | ReLU       | 97.19%     | 10     |
| CNN V2          | Adam      | Tanh       | 84.37%     | 10     |
| CNN V2          | SGD       | Tanh       | *97.31%*   | 10     |
|                 |           |            |            |        |

## Segmentation 

  ### Using handcrafted features
- **Handcrafted features used:**
    1. OTSU thresholding
    2. Canny edge detector
    3. Morphological closing
    4. K means clustering


- Original image used:
![image](https://github.com/user-attachments/assets/6acb70cf-4ce2-451e-863b-4d538b0780ba)

**1. OTSU thresholding:**
  
  - Image was converted to grayscale first.
  - OTSU thresholding was applied using library function provided by opencv.
    ```       
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    ```

    ![image](https://github.com/user-attachments/assets/d8f1695e-2cba-406f-867c-838120f2dd30)

  - **Accuracy**
    Otsu's Thresholding -> IoU: 0.2910, Dice Score: 0.4070
  
**2. Canny edge detector:**
    
  - The image was first converted into grayuscale image.
  - Then library function provided by opencv was used to detect edges
  - This was the worst performing feature among all. 
  
  ```
  edges = cv2.Canny(img, 50, 150)
  ```
  ![image](https://github.com/user-attachments/assets/98ae1ed1-1389-4c3a-a5b7-15708e82db43)

  - **Accuracy**
  Canny Edge Detection -> IoU: 0.1566, Dice Score: 0.2644

**3. Morphological Closing:**
  - Initially the image was converted to grayscale.
  - Then OTSU thresholding was applied to convert it into binary image.
  - Then closing was applied with a kernel of size 3x3.

  ```
    cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
  ```
  ![image](https://github.com/user-attachments/assets/4af2805c-aa7e-49a0-a425-2093a4145c94)

  - **Accuracy**
  Morphological Closing -> IoU: 0.3174, Dice Score: 0.4607

**4. K means clustering:**
  - The K means model was trained for 10 iterations.
  - The final model forms 2 clusters of pixel and assigns the value 0 and 255 to them respectively.
  ```
  _, labels, centers = cv2.kmeans(pixel_values, K, None,
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2), 
                                    10, cv2.KMEANS_RANDOM_CENTERS)
  ```
  ![image](https://github.com/user-attachments/assets/2fd88245-bdea-4cb9-9178-92489734a38d)


  - **Accuracy**
  K means -> IoU: 0.3614, Dice Score: 0.4985


  ### Using UNet:
  - Structure of UNet:
```
def build_light_unet(input_shape=(64, 64, 3)):
    inputs = Input(input_shape)

    # Encoder (Downsampling)
    conv1 = Conv2D(32, (3,3), activation="relu", padding="same")(inputs)
    conv1 = Conv2D(32, (3,3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D((2,2))(conv1)

    conv2 = Conv2D(64, (3,3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(64, (3,3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D((2,2))(conv2)

    conv3 = Conv2D(128, (3,3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(128, (3,3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D((2,2))(conv3)

    # Bottleneck
    conv4 = Conv2D(256, (3,3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(256, (3,3), activation="relu", padding="same")(conv4)

    # Decoder (Upsampling)
    up1 = Conv2DTranspose(128, (2,2), strides=(2,2), padding="same")(conv4)
    up1 = concatenate([up1, conv3])
    conv5 = Conv2D(128, (3,3), activation="relu", padding="same")(up1)
    conv5 = Conv2D(128, (3,3), activation="relu", padding="same")(conv5)

    up2 = Conv2DTranspose(64, (2,2), strides=(2,2), padding="same")(conv5)
    up2 = concatenate([up2, conv2])
    conv6 = Conv2D(64, (3,3), activation="relu", padding="same")(up2)
    conv6 = Conv2D(64, (3,3), activation="relu", padding="same")(conv6)

    up3 = Conv2DTranspose(32, (2,2), strides=(2,2), padding="same")(conv6)
    up3 = concatenate([up3, conv1])
    conv7 = Conv2D(32, (3,3), activation="relu", padding="same")(up3)
    conv7 = Conv2D(32, (3,3), activation="relu", padding="same")(conv7)

    # Output layer
    outputs = Conv2D(1, (1,1), activation="sigmoid")(conv7)

    # Compile Model
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model
```
![image](https://github.com/user-attachments/assets/76cb56c7-5a34-47ea-a8e3-f4ca4fbdbe77)

- Total params: 1,925,601 (7.35 MB)\
  Trainable params: 1,925,601 (7.35 MB)\
  Non-trainable params: 0 (0.00 B)\

![image](https://github.com/user-attachments/assets/4a556e41-3687-4ef5-80e0-5521320e88f7)

 - **Accuracy**
val_accuracy: 0.9712 - val_loss: 0.0731\
Test Accuracy: 97.12%

#### Hyperparameters and Experiments:
- **Batch Size:** 32
- **Epochs:** 50,100
- **Learning Rate:** 0.001
- **Optimizer:** Adam
- **Loss Function:** Binary Cross-Entropy
- **Metrics:** Accuracy
- **Activation Functions:** ReLU, Sigmoid
- **Model Architecture:** U-Net with Conv2D and Conv2DTranspose layers
- **Number of Kernels:** 32,64,128,256,512
- **Kernel Size:** 3,5,7
- **Stride:** 1,2,3


# Observations and Analysis:
In both the Tasks it is observed that the handcrafted features are performing poorly when compared to the respective CNN architectures.
  






