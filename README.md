# TOPIC NAME
Machine learning model to classify iris flowers

# ABOUT THE TOPIC PROBLEM STATEMENT
The Iris flower classification project is a classic example of supervised machine learning, where the goal is to classify iris flowers into three species—Setosa, Versicolor, and Virginica—based on their physical measurements. This project utilizes the well-known Iris dataset, which contains measurements of sepal length, sepal width, petal length, and petal width for 150 iris flowers.

# FUNCTIONAL COMPONENTS
• Loading the dataset using sklearn or pandas
• Data preparation: feature and label separation
• Train-test splitting using train_test_split()
• Model training using DecisionTreeClassifier()
• Model evaluation using accuracy score
• Predicting output for new flower data

# Output
Accuracy: 0.98
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        19
Iris-versicolor       0.94      1.00      0.97        17
 Iris-virginica       1.00      0.94      0.97        17

       accuracy                           0.98        53
      macro avg       0.98      0.98      0.98        53
   weighted avg       0.98      0.98      0.98        53
