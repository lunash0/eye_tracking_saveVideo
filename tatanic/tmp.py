import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train_file = 'C:/Users/82108/PycharmProjects/eye_tracking_savedVideo_0/tatanic/train.csv'
test_file = 'C:/Users/82108/PycharmProjects/eye_tracking_savedVideo_0/tatanic/test.csv'
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
print(train.columns)
train.head()

test.insert(loc=1, column='Survived', value=0)
test_tail = test.tail()
print(test_tail)

data = pd.concat([train, test], axis=0)
data_tail = data.tail()
print(data_tail)

data = data.dropna(thresh=int(len(train) * 0.5), axis=1)

age_mean = data['Age'].mean()
data['Age'] = data['Age'].fillna(age_mean)

fare_median = data['Fare'].median()
data['Fare'] = data['Fare'].fillna(fare_median)

data['Embarked'] = data['Embarked'].fillna('S')

passenger_id = data[['PassengerId']].copy()
data = data.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

data = pd.get_dummies(data)

train, test = data[:len(train)], data[len(train):]

trainX = train[[x for x in train.columns if 'Survived' not in x]]
trainY = train[['Survived']]

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same', input_shape=(trainX.shape[1],)),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(trainX, trainY, epochs=10, batch_size=150)

result = passenger_id[len(train):]

testX = np.expand_dims(test.values, axis=-1)
testX = np.concatenate([testX] * 3, axis=-1)
testX = testX.reshape(-1, 32, 32, 3)
testY = np.round(model.predict(testX))
testY = pd.DataFrame(testY, columns=['Survived'], dtype=int)
result = pd.concat([result, testY], axis=1)
result = result.set_index('PassengerId')
