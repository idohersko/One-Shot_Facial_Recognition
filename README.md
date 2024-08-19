# One-Shot Facial Recognition with Siamese Neural Networks

![image](https://github.com/user-attachments/assets/362a804a-b61b-4e8f-8a60-42fc29375627)


This project focuses on implementing a convolutional neural network (CNN) to perform one-shot facial recognition. Inspired by the ["Siamese Neural Networks for One-shot Image Recognition"](https://www.cs.cmu.edu/%7Ersalakhu/papers/oneshot1.pdf) paper, the goal is to develop a model capable of determining whether two facial images belong to the same person, even when the faces are previously unseen.

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Setup and Experiments](#setup-and-experiments)
5. [Performance and Results](#performance-and-results)
6. [Conclusion](#conclusion)

## Overview

In this project, we implement a Siamese Neural Network to solve the problem of one-shot facial recognition. The model is designed to learn a similarity metric that can distinguish between images of the same person and different people based on minimal training data. The core architecture is built using TensorFlow and follows the guidelines provided by Koch et al. in their 2015 paper.

## Dataset

The project uses the "Labeled Faces in the Wild-a" dataset, which includes labeled pairs of facial images. The dataset is split into training and testing sets, ensuring no overlap between the two. The training set consists of 2200 pairs (1100 matching and 1100 mismatching), while the test set includes 1000 pairs (500 matching and 500 mismatching). The data was further split into training and validation sets in an 85:15 ratio to monitor the model's performance and prevent overfitting.

![image](https://github.com/user-attachments/assets/1a00786e-8a86-4bd7-a5a4-bc152da83abd)


![image](https://github.com/user-attachments/assets/4ba949b1-312e-46fb-bf51-61ab62343818)  ![image](https://github.com/user-attachments/assets/202bf042-5182-4a4a-8146-c3dd36622f60)

![image](https://github.com/user-attachments/assets/8a75f119-f2bc-41f6-b645-c0459fdb47da)

### Preprocessing

- Images are resized to 105x105 pixels, following the dimensions used in the original paper.
- Data is normalized and labeled as matching (1) or mismatching (0) based on the pair type.

## Model Architecture


![image](https://github.com/user-attachments/assets/e27ff231-66eb-4b8d-bfbf-ada2032d995b)

![image](https://github.com/user-attachments/assets/aac720d6-bb06-4ab5-8176-cd7058a629a6) ![image](https://github.com/user-attachments/assets/3c500b56-23d5-4a3a-8eef-e0f9fdb383b8)


The model is based on a Siamese Neural Network with the following architecture:

- **Input Layers:** Two input layers, each receiving an image of size 105x105.
- **CNN Feature Extractor:** Four convolutional layers with batch normalization, ReLU activation, and max-pooling:
  - Convolutional layers with 64, 128, 128, and 256 filters.
  - Fully connected layer with 4096 neurons using a sigmoid activation function.
- **Distance Calculation:** The L1 distance metric is used to compare the feature vectors from the two input images.
- **Output Layer:** A fully connected layer with a sigmoid activation function to predict whether the images match or not.

### Model Visualization

The model architecture was visualized and adjusted to suit the facial recognition task, following the principles outlined in the referenced paper.

## Setup and Experiments

The project was conducted using Python 3.9 and TensorFlow 2.0. A series of experiments were performed with varying hyperparameters to optimize the model's performance. Key configurations included:

- **Model 1:** 200 epochs, batch size of 8, SGD optimizer, learning rate 0.01.
- **Model 2:** 200 epochs, batch size of 32, Adam optimizer, learning rate 0.001.
- **Model 3:** 100 epochs, batch size of 8, Adam optimizer, learning rate 0.0001.

Each model was evaluated using binary accuracy, AUC, and loss metrics. Early stopping with patience set to 20 epochs was implemented to prevent overfitting.

 **Model 1's graphs :**

![image](https://github.com/user-attachments/assets/d2c8385c-d5ae-4d86-a8ed-6f974294e2d8)

 **Model 2's graphs :**

![image](https://github.com/user-attachments/assets/7e5a0380-122a-46f4-ad4a-071da3076c1c)

 **Model 3's graphs :**

![image](https://github.com/user-attachments/assets/64d74e5a-802d-42e0-8096-48c2f48fdbdf)

 
### Grid Search

A grid search was performed to identify the best combination of hyperparameters. 36 models were trained, varying the number of epochs, learning rates, optimizers, and batch sizes. The best-performing model was selected based on validation accuracy.
![image](https://github.com/user-attachments/assets/2dc96572-c115-48bf-9156-0c753a248c3c)

| Numb | Epoch | LR     | Batch | Optimizer | Train Loss | Train Accuracy | Train AUC | Validation Loss | Validation Accuracy | Validation AUC | Time      |
|------|-------|--------|-------|-----------|------------|----------------|-----------|-----------------|---------------------|----------------|-----------|
| 1    | 100   | 0.0001 | 8     | Adam      | 0.2128     | 0.9706         | 0.9973    | 0.6450          | 0.6920              | 0.7320         | 23.62 min |
| 2    | 100   | 0.001  | 8     | Adam      | 0.6265     | 0.6989         | 0.7835    | 0.6563          | 0.6470              | 0.7397         | 23.68 min |
| 3    | 100   | 0.01   | 8     | Adam      | 0.8050     | 0.5139         | 0.5167    | 0.8050          | 0.5130              | 0.5153         | 35.6 min  |
| 4    | 200   | 0.0001 | 8     | Adam      | 0.1854     | 0.9840         | 0.9992    | 0.6346          | 0.6680              | 0.7377         | 23.75 min |
| 5    | 200   | 0.001  | 8     | Adam      | 0.6745     | 0.6364         | 0.7203    | 0.6766          | 0.6060              | 0.7133         | 23.78 min |
| 6    | 200   | 0.01   | 8     | Adam      | 0.7909     | 0.5011         | 0.5165    | 0.7901          | 0.5010              | 0.5281         | 40.5 min  |
| 7    | 100   | 0.0001 | 16    | Adam      | 0.1867     | 0.9888         | 0.9995    | 0.6365          | 0.6740              | 0.7379         | 20.67 min |
| 8    | 100   | 0.001  | 16    | Adam      | 0.5221     | 0.7765         | 0.8664    | 0.6530          | 0.6610              | 0.7265         | 20.37 min |
| 9    | 100   | 0.01   | 16    | Adam      | 0.7678     | 0.5294         | 0.5289    | 0.7661          | 0.5310              | 0.5553         | 29.52 min |
| 10   | 200   | 0.0001 | 16    | Adam      | 0.1705     | 0.9920         | 0.9999    | 0.6282          | 0.6900              | 0.7429         | 20.22 min |
| 11   | 200   | 0.001  | 16    | Adam      | 0.5622     | 0.7503         | 0.8290    | 0.6710          | 0.6500              | 0.7017         | 20.6 min  |
| 12   | 200   | 0.01   | 16    | Adam      | 0.8350     | 0.5733         | 0.6024    | 0.8234          | 0.5900              | 0.6314         | 30.58 min |
| 13   | 100   | 0.0001 | 32    | Adam      | 0.1794     | 0.9973         | 1.0000    | 0.6401          | 0.6550              | 0.7221         | 18.37 min |
| 14   | 100   | 0.001  | 32    | Adam      | 0.3988     | 0.8786         | 0.9518    | 0.6821          | 0.6590              | 0.6984         | 18.28 min |
| 15   | 100   | 0.01   | 32    | Adam      | 0.8237     | 0.5385         | 0.5700    | 0.8098          | 0.5940              | 0.6275         | 30.28 min |
| 16   | 200   | 0.0001 | 32    | Adam      | 0.1615     | 0.9984         | 1.0000    | 0.6419          | 0.6540              | 0.7177         | 18.17 min |
| 17   | 200   | 0.001  | 32    | Adam      | 0.4110     | 0.8658         | 0.9415    | 0.6789          | 0.6760              | 0.7096         | 18.67 min |
| 18   | 200   | 0.01   | 32    | Adam      | 0.8334     | 0.5583         | 0.5707    | 0.8360          | 0.5350              | 0.5578         | 33.82 min |
| 19   | 100   | 0.0001 | 8     | SGD       | 0.4922     | 0.8610         | 0.9402    | 0.7121          | 0.5790              | 0.5942         | 14.99 min |
| 20   | 100   | 0.001  | 8     | SGD       | 0.3479     | 0.9406         | 0.9886    | 0.6555          | 0.6240              | 0.7015         | 15.18 min |
| 21   | 100   | 0.01   | 8     | SGD       | 0.2269     | 0.9369         | 0.9871    | 0.7437          | 0.6380              | 0.6911         | 20.02 min |
| 22   | 200   | 0.0001 | 8     | SGD       | 0.4759     | 0.8872         | 0.9559    | 0.7012          | 0.5790              | 0.6143         | 15.1 min  |
| 23   | 200   | 0.001  | 8     | SGD       | 0.3800     | 0.9289         | 0.9798    | 0.6623          | 0.6420              | 0.6832         | 15.02 min |
| 24   | 200   | 0.01   | 8     | SGD       | 0.2336     | 0.9460         | 0.9875    | 0.6928          | 0.6790              | 0.7303         | 15.17 min |
| 25   | 100   | 0.0001 | 16    | SGD       | 0.5152     | 0.8460         | 0.9245    | 0.7344          | 0.5250              | 0.5422         | 13.28 min |
| 26   | 100   | 0.001  | 16    | SGD       | 0.3618     | 0.9487         | 0.9897    | 0.6723          | 0.6300              | 0.6804         | 13.15 min |
| 27   | 100   | 0.01   | 16    | SGD       | 0.2492     | 0.9583         | 0.9923    | 0.6800          | 0.6370              | 0.6999         | 13.39 min |
| 28   | 200   | 0.0001 | 16    | SGD       | 0.4990     | 0.8695         | 0.9428    | 0.7344          | 0.5330              | 0.5428         | 13.45 min |
| 29   | 200   | 0.001  | 16    | SGD       | 0.3569     | 0.9412         | 0.9870    | 0.6693          | 0.6330              | 0.6775         | 13.39 min |
| 30   | 200   | 0.01   | 16    | SGD       | 0.2671     | 0.9524         | 0.9902    | 0.6881          | 0.6190              | 0.6915         | 13.46 min |
| 31   | 100   | 0.0001 | 32    | SGD       | 0.5340     | 0.8257         | 0.9123    | 0.7536          | 0.4810              | 0.5015         | 12.41 min |
| 32   | 100   | 0.001  | 32    | SGD       | 0.3862     | 0.9465         | 0.9897    | 0.7030          | 0.5780              | 0.6149         | 12.38 min |
| 33   | 100   | 0.01   | 32    | SGD       | 0.2525     | 0.9711         | 0.9974    | 0.6647          | 0.6370              | 0.6928         | 12.36 min |
| 34   | 200   | 0.0001 | 32    | SGD       | 0.5415     | 0.8230         | 0.9043    | 0.7425          | 0.5200              | 0.5210         | 12.57 min |
| 35   | 200   | 0.001  | 32    | SGD       | 0.3951     | 0.9449         | 0.9871    | 0.6848          | 0.6030              | 0.6438         | 12.63 min |
| 36   | 200   | 0.01   | 32    | SGD       | 0.3006     | 0.9545         | 0.9908    | 0.6604          | 0.6310              | 0.6955         | 12.52 min |
| AVG  |       |        |       |           | 0.4434     | 0.8398         | 0.8844    | 0.7008          | 0.6101              | 0.6559         | 21.5 min  |

### Final Model Configuration

The final model was trained with the following setup:
- **Epochs:** 100 (with early stopping, patience = 5)
- **Batch Size:** 8
- **Optimizer:** Adam
- **Learning Rate:** 0.0001
- **Loss Function:** Binary Crossentropy
- **Dropout:** 0.5 (added to reduce overfitting)

![image](https://github.com/user-attachments/assets/cbe5478b-9cc7-4669-9281-30aa94c904c1)
![image](https://github.com/user-attachments/assets/5d3ec7b9-f19d-4b51-b4bd-93858de663bf)


## Performance and Results

The final model demonstrated robust performance, achieving a good balance between training accuracy and generalization on the validation set. However, the model struggled with certain cases, particularly when images included accessories or significant background noise, indicating areas for potential improvement.

### Prediction Examples

The project includes examples of both the best and worst predictions made by the model. Common issues included misclassification due to background elements or accessories like sunglasses or hats.

![image](https://github.com/user-attachments/assets/3f360a06-7f51-42a4-8655-f6591fe0de54)  ![image](https://github.com/user-attachments/assets/d8b708fa-353b-48e0-ae12-794d2e25b471)




## Conclusion

This project provided a deep dive into one-shot learning and facial recognition using Siamese Neural Networks. Through extensive experimentation, it was evident that careful tuning of hyperparameters, combined with dropout for regularization, was crucial in achieving optimal performance. The final model serves as a strong foundation for further exploration and application of CNNs in real-world facial recognition tasks.

## Authors

- Elran Oren
- Ido Hersko
