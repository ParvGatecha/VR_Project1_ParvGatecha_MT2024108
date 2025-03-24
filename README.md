# VR Assignment 

## Objectives:
- To develop a computer vision solution to classify and segment face masks in images.
- The project involves using handcrafted features with machine learning classifiers and deep
learning techniques to perform classification and segmentation.


## Dataset:
- A labeled dataset containing images of people with and without face masks: https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset

- A Masked Face Segmentation Dataset with ground truth face masks can be accessed
here: https://github.com/sadjadrz/MFSD

# Classification
## Methodology:

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


# Segmentation 

