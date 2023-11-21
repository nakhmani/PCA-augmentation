```python
from sklearn.utils import shuffle
#from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn import metrics

#data handling
from keras.backend import dropout
import pandas as pd
import numpy as np

#data visualization
import matplotlib.pyplot as plt
import seaborn as sns

#preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

#classification
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
```


```python
# Load data1 and data2
data1 = pd.read_csv('G:/PCA/heart.csv')
data2 = pd.read_csv('G:/PCA/7030heart-151.csv')

#Split data1 into class=N and class=C
data1_Y = data1[data1['target'] == 1]
data1_N = data1[data1['target'] == 0]

#Split class=N and class=C data1 into training and testing sets
train_data1_Y, test_data1_Y = train_test_split(data1_Y, test_size=0.3)
train_data1_N, test_data1_N = train_test_split(data1_N, test_size=0.3)

#Merge class=N training data, class=C training data, and data2
train_data = pd.concat([train_data1_Y, train_data1_N, data2], axis=0, ignore_index=True)

#Use merged training data for training
X_train = train_data.drop('target', axis=1)
y_train = train_data['target']

#Use remaining data1 for testing
test_data1 = pd.concat([test_data1_Y, test_data1_N], axis=0, ignore_index=True)
X_test = test_data1.drop('target', axis=1)
y_test = test_data1['target']
```


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
```


```python
#let's encode target labels (y) with values between 0 and n_classes-1.
#encoding will be done using the LabelEncoder
label_encoder=LabelEncoder()
label_encoder.fit(y_train)
y_train=label_encoder.transform(y_train)
labels=label_encoder.classes_
classes=np.unique(y_train)
nclasses=np.unique(y_train).shape[0]
```


```python
#let's encode target labels (y) with values between 0 and n_classes-1.
#encoding will be done using the LabelEncoder
label_encoder.fit(y_test)
y_test=label_encoder.transform(y_test)
labels2=label_encoder.classes_
classes2=np.unique(y_test)
nclasses2=np.unique(y_test).shape[0]
```


```python
#split data into training,validation and test sets

#split the training set into two (training and validation)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.3)
```


```python
### scale the data between 0-1
min_max_scaler=MinMaxScaler()
X_train=min_max_scaler.fit_transform(X_train)
X_val=min_max_scaler.fit_transform(X_val)
X_test=min_max_scaler.fit_transform(X_test)
```


```python
accuracies = []
models = []

for i in range(10):
    # Define and compile the model
    #model = Sequential()
    #model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
    #model.add(Dense(1, activation='sigmoid'))
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model = tf.keras.Sequential([
    Dense(512, activation='relu', input_dim=13),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Shuffle the training data and labels
    X_train, y_train = shuffle(X_train, y_train)

    # Fit the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)

    # Evaluate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

     # Evaluate the model on the test data
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Append the model to the list of models
    models.append(model)
    
print(accuracies)
```

    [0.9253246753246753, 0.9285714285714286, 0.9188311688311688, 0.9383116883116883, 0.9188311688311688, 0.9188311688311688, 0.9253246753246753, 0.9188311688311688, 0.935064935064935, 0.9253246753246753]
    


```python
mean_accuracy = sum(accuracies) / len(accuracies)
print("Mean accuracy:", mean_accuracy)
```

    Mean accuracy: 0.9253246753246753
    


```python
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Get the index of the best accuracy
best_accuracy_index = np.argmax(accuracies)

# Get the best model based on the index
best_model = models[best_accuracy_index]

# Use the best model to make predictions on the test data
y_pred = best_model.predict(X_test)

# Round the predictions to get binary values (0 or 1)
y_pred = np.round(y_pred)

# Calculate the confusion matrix
#
conf_matrix = confusion_matrix(y_test, y_pred)
#
# Print the confusion matrix using Matplotlib
#
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=16)
plt.ylabel('Actuals', fontsize=16)
plt.title('Confusion Matrix', fontsize=16)

# Add labels to the sections of the confusion matrix
#ax.text(0, 0, "True Negatives\n(TN):\n" + str(TN), ha='center', va='center', color='black')
#ax.text(1, 0, "False Negatives\n(FN):\n" + str(FN), ha='center', va='center', color='black')
#ax.text(0, 1, "False Positives\n(FP):\n" + str(FP), ha='center', va='center', color='black')
#ax.text(1, 1, "True Positives\n(TP):\n" + str(TP), ha='center', va='center', color='black')

plt.show()

from sklearn.metrics import confusion_matrix

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Extract TP, TN, FP, FN values
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

# Calculate sensitivity and specificity
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

# Print results
print("Confusion Matrix:")
print(cm)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

```


    
![png](output_9_0.png)
    


    Confusion Matrix:
    [[136  14]
     [  5 153]]
    Sensitivity: 0.9683544303797469
    Specificity: 0.9066666666666666
    


```python
best_accuracy = max(accuracies)
best_model_index = accuracies.index(best_accuracy)
best_model = models[best_model_index]

# Make predictions on the test data using the best model
y_pred = best_model.predict(X_test)
y_pred = [round(x[0]) for x in y_pred]

# Compare the actual and predicted values
for i in range(len(y_test)):
    print("Actual value:", y_test[i], "Predicted value:", y_pred[i])
```

    Actual value: 1 Predicted value: 0
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 0
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 0
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 0
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 0
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    


```python
import matplotlib.pyplot as plt

history = best_model.fit(X_train, y_train, batch_size = 32, verbose = 0, epochs = 100, validation_data=(X_val, y_val),
                    shuffle = False)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model performance')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')
plt.show()
```


    
![png](output_11_0.png)
    



```python
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training loss', 'validation loss'], loc='lower right')
plt.show()
```


    
![png](output_12_0.png)
    


Logistic Regression Model


```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
clf = LogisticRegression(max_iter=1000)
```


```python
# Fit the logistic regression model to the training data
clf = LogisticRegression()
clf.fit(X_train, y_train)
```




    LogisticRegression()




```python
# Use the model to make predictions on the test data
y_pred1 = clf.predict(X_test)

# Compare the actual and predicted values
for i in range(len(y_test)):
    print("Actual value:", y_test[i], "Predicted value:", y_pred1[i])
```

    Actual value: 1 Predicted value: 0
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 0
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 0
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 0
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 0
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 0
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 0
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 0
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 0
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 0
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    


```python
# Calculate the accuracy of the model
acc = accuracy_score(y_test, y_pred1)
print("Accuracy: ", acc)
```

    Accuracy:  0.8538961038961039
    


```python
# Repeat the process 10 times and store the accuracy in a list
accuracies1 = []
for i in range(10):
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)
    clf.fit(X_train, y_train)
    y_pred1 = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred1)
    accuracies1.append(acc)

print(accuracies1)
```

    [0.8538961038961039, 0.8538961038961039, 0.8538961038961039, 0.8538961038961039, 0.8538961038961039, 0.8538961038961039, 0.8538961038961039, 0.8538961038961039, 0.8538961038961039, 0.8538961038961039]
    


```python
# Calculate the mean accuracy over the 10 iterations
mean_accuracy1 = np.mean(accuracies1)
print("Mean accuracy: ", mean_accuracy1)
```

    Mean accuracy:  0.8538961038961039
    


```python
plot_confusion_matrix(clf, X_train, y_train)
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x267f7c3ee20>




    
![png](output_20_1.png)
    



```python
plot_confusion_matrix(clf, X_test, y_test)

from sklearn.metrics import confusion_matrix

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred1)

# Extract TP, TN, FP, FN values
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

# Calculate sensitivity and specificity
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

# Print results
print("Confusion Matrix:")
print(cm)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

```

    Confusion Matrix:
    [[115  35]
     [ 10 148]]
    Sensitivity: 0.9367088607594937
    Specificity: 0.7666666666666667
    


    
![png](output_21_1.png)
    


Support Vector Machine


```python
from sklearn.svm import SVC
```


```python
# Create a SVM classifier with linear kernel
svm = SVC(kernel='linear')

# Train the SVM classifier
svm.fit(X_train, y_train)

# Predict on test data
y_pred2 = svm.predict(X_test)

# Compare the actual and predicted values
for i in range(len(y_test)):
    print("Actual value:", y_test[i], "Predicted value:", y_pred2[i])
```

    Actual value: 1 Predicted value: 0
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 0
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 0
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 0
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 0
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 0
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 0
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 0
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 1 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    Actual value: 0 Predicted value: 0
    Actual value: 0 Predicted value: 1
    


```python
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred2)

print("Accuracy:", accuracy)
```

    Accuracy: 0.8409090909090909
    


```python
# initialize list to store accuracies
accuracies = []

# repeat the process 10 times
for i in range(10):
    # split the data into train and test sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # create SVM model
    model = SVC(kernel='linear', C=1)
    
    # fit the model on training data
    model.fit(X_train, y_train)
    
    # make predictions on test data
    y_pred = model.predict(X_test)
    
    # calculate accuracy and append it to list of accuracies
    acc = accuracy_score(y_test, y_pred2)
    accuracies.append(acc)

# print list of accuracies and mean accuracy
print("List of accuracies:", accuracies)
print("Mean accuracy:", np.mean(accuracies))
```

    List of accuracies: [0.8409090909090909, 0.8409090909090909, 0.8409090909090909, 0.8409090909090909, 0.8409090909090909, 0.8409090909090909, 0.8409090909090909, 0.8409090909090909, 0.8409090909090909, 0.8409090909090909]
    Mean accuracy: 0.840909090909091
    


```python
# Calculate confusion matrix
plot_confusion_matrix(svm,X_test, y_test)


from sklearn.metrics import confusion_matrix

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred2)

# Extract TP, TN, FP, FN values
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

# Calculate sensitivity and specificity
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

# Print results
print("Confusion Matrix:")
print(cm)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)


```

    Confusion Matrix:
    [[109  41]
     [  8 150]]
    Sensitivity: 0.9493670886075949
    Specificity: 0.7266666666666667
    


    
![png](output_27_1.png)
    


SVM Poly kerner degree 2


```python
from sklearn.svm import SVC
from sklearn import svm
```


```python
# Create a SVM classifier with poly kernel
svm2 = svm.SVC(kernel='poly', degree=2)

# Train the SVM classifier
svm2.fit(X_train, y_train)

# Predict on test data
y_pred3 = svm2.predict(X_test)

# Compare the actual and predicted values
#for i in range(len(y_test)):
#    print("Actual value:", y_test[i], "Predicted value:", y_pred3[i])
```


```python
# initialize list to store accuracies
accuracies = []

# repeat the process 10 times
for i in range(10):
    # split the data into train and test sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # create SVM model
    model = svm.SVC(kernel='poly', degree=2)
    
    # fit the model on training data
    model.fit(X_train, y_train)
    
    # make predictions on test data
    y_pred = model.predict(X_test)
    
    # calculate accuracy and append it to list of accuracies
    acc = accuracy_score(y_test, y_pred3)
    accuracies.append(acc)

# print list of accuracies and mean accuracy
print("List of accuracies:", accuracies)
print("Mean accuracy:", np.mean(accuracies))
```

    List of accuracies: [0.8506493506493507, 0.8506493506493507, 0.8506493506493507, 0.8506493506493507, 0.8506493506493507, 0.8506493506493507, 0.8506493506493507, 0.8506493506493507, 0.8506493506493507, 0.8506493506493507]
    Mean accuracy: 0.8506493506493505
    


```python
# Calculate confusion matrix
plot_confusion_matrix(svm2,X_test, y_test)


from sklearn.metrics import confusion_matrix

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred3)

# Extract TP, TN, FP, FN values
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

# Calculate sensitivity and specificity
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

# Print results
print("Confusion Matrix:")
print(cm)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
```

    Confusion Matrix:
    [[111  39]
     [  7 151]]
    Sensitivity: 0.9556962025316456
    Specificity: 0.74
    


    
![png](output_32_1.png)
    

