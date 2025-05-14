# SMS Spam Classifier using Perceptron

This project implements a perceptron-based spam classifier trained on the [SMS Spam Collection dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset). The goal was to build a binary classifier from scratch using the classical perceptron algorithm and evaluate its performance on real-world text data.

## Dataset
The dataset contains 5,572 SMS messages labeled as either:
- `ham` (not spam)
- `spam`

We used `CountVectorizer` to convert each message into a bag-of-words representation.

## Perceptron
- Binary labels were encoded as `+1` (spam) and `-1` (ham)
- We added a bias term to each input vector
- The perceptron was trained over 10 epochs using the standard update rule:
  
```Python
if y_i * (w ⋅ x_i) <= 0:
w += learning_rate * y_i * x_i
```


## Results

After training the model for 10 epochs, we evaluated it on a 20% test split (stratified to preserve label balance). Here's what we got:

              precision    recall  f1-score   support

          -1       0.98      0.99      0.99       966
           1       0.96      0.89      0.93       149

    accuracy                           0.98      1115
   macro avg       0.97      0.94      0.96
weighted avg       0.98      0.98      0.98


## Key Takeaways

- The model correctly identifies most spam messages while keeping false positives low.
- Even without any deep learning or libraries like `scikit-learn` for the model itself, the classic perceptron performs surprisingly well on this task.