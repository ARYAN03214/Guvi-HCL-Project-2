{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "XFh6eJT5sP_K",
        "outputId": "e7c7105c-4697-4a0d-9809-08ab31bf3f83"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy: 0.98\n",
            "                 precision    recall  f1-score   support\n",
            "\n",
            "    Iris-setosa       1.00      1.00      1.00        19\n",
            "Iris-versicolor       0.94      1.00      0.97        17\n",
            " Iris-virginica       1.00      0.94      0.97        17\n",
            "\n",
            "       accuracy                           0.98        53\n",
            "      macro avg       0.98      0.98      0.98        53\n",
            "   weighted avg       0.98      0.98      0.98        53\n",
            "\n",
            "The predicted species for the input [[5.0, 3.4, 1.5, 0.2]] is: Iris-setosa\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py:2739: UserWarning: X does not have valid feature names, but DecisionTreeClassifier was fitted with feature names\n",
            "  warnings.warn(\n"
          ]
        }
      ],
      "source": [
        "# Import necessary libraries\n",
        "import pandas as pd\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn.metrics import accuracy_score, classification_report\n",
        "\n",
        "# Load the dataset from a CSV file\n",
        "data = pd.read_csv('//content/datairis.csv')\n",
        "\n",
        "# Prepare features and target\n",
        "X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]\n",
        "y = data['Species']\n",
        "\n",
        "# Split the dataset into training and testing subsets\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)\n",
        "\n",
        "# Train a Decision Tree Classifier\n",
        "model = DecisionTreeClassifier()\n",
        "model.fit(X_train, y_train)\n",
        "\n",
        "# Predict species for the test set\n",
        "y_pred = model.predict(X_test)\n",
        "\n",
        "# Evaluate accuracy\n",
        "accuracy = accuracy_score(y_test, y_pred)\n",
        "print(f'Accuracy: {accuracy:.2f}')\n",
        "\n",
        "# Display classification report\n",
        "print(classification_report(y_test, y_pred))\n",
        "\n",
        "# Predicting output for new flower data\n",
        "new_flower = [[5.0, 3.4, 1.5, 0.2]]\n",
        "predicted_species = model.predict(new_flower)\n",
        "print(f'The predicted species for the input {new_flower} is: {predicted_species[0]}')\n"
      ]
    }
  ]
}