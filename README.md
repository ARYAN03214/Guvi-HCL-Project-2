🌸 Iris Flower Classification using Machine Learning

📌 Problem Statement

The Iris Flower Classification project is a classic example of supervised machine learning.
The goal is to classify iris flowers into three species:

🌿 Setosa

🌼 Versicolor

🌸 Virginica

based on their physical measurements:

•Sepal Length

•Sepal Width

•Petal Length

•Petal Width

The project uses the famous Iris dataset (150 samples) to train and test the model.

⚙️ Functional Components

✅ Load dataset using sklearn / pandas

✅ Prepare features & labels

✅ Split data into training and testing sets (train_test_split)

✅ Train model with DecisionTreeClassifier

✅ Evaluate using accuracy score & classification report

✅ Predict species for new flower inputs

📊 Model Performance

The trained Decision Tree Classifier achieved an impressive accuracy of 98% 🎯.

 Accuracy: 0.98
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        19
Iris-versicolor       0.94      1.00      0.97        17
 Iris-virginica       1.00      0.94      0.97        17

       accuracy                           0.98        53
      macro avg       0.98      0.98      0.98        53
   weighted avg       0.98      0.98      0.98        53

🚀 Output Example

# Example input
new_data = [[5.0, 3.4, 1.5, 0.2]]

# Predicted output
🌸 Predicted Species: Iris-setosa
