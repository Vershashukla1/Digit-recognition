# %% [code] {"execution":{"iopub.status.busy":"2023-11-21T06:32:32.255775Z","iopub.execute_input":"2023-11-21T06:32:32.256050Z","iopub.status.idle":"2023-11-21T06:32:44.325552Z","shell.execute_reply.started":"2023-11-21T06:32:32.256025Z","shell.execute_reply":"2023-11-21T06:32:44.324767Z"}}
import pandas as pd
import numpy as np
import tensorflow as tf
import csv
from tensorflow.keras import layers

# %% [code] {"execution":{"iopub.status.busy":"2023-11-21T06:32:44.326764Z","iopub.execute_input":"2023-11-21T06:32:44.327296Z","iopub.status.idle":"2023-11-21T06:32:44.333713Z","shell.execute_reply.started":"2023-11-21T06:32:44.327268Z","shell.execute_reply":"2023-11-21T06:32:44.332698Z"}}
def read_csv_file(file_path):
    images = []
    labels = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            label = int(row[0])
            pixels = [int(x) for x in row[1:]]
            images.append(pixels)
            labels.append(label)
    return images, labels

# %% [code] {"execution":{"iopub.status.busy":"2023-11-21T06:32:44.335193Z","iopub.execute_input":"2023-11-21T06:32:44.335860Z","iopub.status.idle":"2023-11-21T06:32:52.802287Z","shell.execute_reply.started":"2023-11-21T06:32:44.335826Z","shell.execute_reply":"2023-11-21T06:32:52.801535Z"}}
# Read the train.csv file
train_images, train_labels = read_csv_file("C:\\Users\\versha.shukla\\Desktop\\project\\digit-recognizer\\train.csv")

# Print the first image and label
print(train_labels[0])  # Label of the first image
print(train_images[0])  # Pixel values list of the first image
# %% [code] {"execution":{"iopub.status.busy":"2023-11-21T06:32:52.805600Z","iopub.execute_input":"2023-11-21T06:32:52.806218Z","iopub.status.idle":"2023-11-21T06:32:52.811542Z","shell.execute_reply.started":"2023-11-21T06:32:52.806191Z","shell.execute_reply":"2023-11-21T06:32:52.810670Z"}}
def read_testcsv_file(file_path):
    images = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader) #skip the header row
        for row in csv_reader:
            pixels = [int(x) for x in row[0:]]
            images.append(pixels)
    return images

# %% [code] {"execution":{"iopub.status.busy":"2023-11-21T06:32:52.812811Z","iopub.execute_input":"2023-11-21T06:32:52.813148Z","iopub.status.idle":"2023-11-21T06:32:58.174856Z","shell.execute_reply.started":"2023-11-21T06:32:52.813115Z","shell.execute_reply":"2023-11-21T06:32:58.174059Z"}}
# Read test dataset images
test_images = read_testcsv_file("C:\\Users\\versha.shukla\\Desktop\\project\\digit-recognizer\\test.csv")

# %% [code] {"execution":{"iopub.status.busy":"2023-11-21T06:32:58.175894Z","iopub.execute_input":"2023-11-21T06:32:58.176161Z","iopub.status.idle":"2023-11-21T06:33:03.319144Z","shell.execute_reply.started":"2023-11-21T06:32:58.176138Z","shell.execute_reply":"2023-11-21T06:33:03.318361Z"}}
# Normalize image data
train_images = np.array(train_images)
test_images = np.array(test_images)
# Normalize image data to range between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# %% [code] {"execution":{"iopub.status.busy":"2023-11-21T06:33:03.320280Z","iopub.execute_input":"2023-11-21T06:33:03.320584Z","iopub.status.idle":"2023-11-21T06:33:03.328521Z","shell.execute_reply.started":"2023-11-21T06:33:03.320561Z","shell.execute_reply":"2023-11-21T06:33:03.327649Z"}}
train_labels = np.array(train_labels)

# %% [code] {"execution":{"iopub.status.busy":"2023-11-21T06:33:03.329651Z","iopub.execute_input":"2023-11-21T06:33:03.329985Z","iopub.status.idle":"2023-11-21T06:33:03.740493Z","shell.execute_reply.started":"2023-11-21T06:33:03.329954Z","shell.execute_reply":"2023-11-21T06:33:03.739520Z"}}
from sklearn.model_selection import train_test_split

# Split the training dataset into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Print the sizes of the training and validation sets
print("Training set size:", len(train_images))
print("Validation set size:", len(val_images))

# %% [code] {"execution":{"iopub.status.busy":"2023-11-21T06:33:03.741717Z","iopub.execute_input":"2023-11-21T06:33:03.742025Z","iopub.status.idle":"2023-11-21T06:33:06.873000Z","shell.execute_reply.started":"2023-11-21T06:33:03.741999Z","shell.execute_reply":"2023-11-21T06:33:06.872111Z"}}
# Define CNN model
model = tf.keras.Sequential([
    layers.Reshape((28, 28, 1), input_shape=(784,)),  
    layers.Conv2D(32, (3, 3), activation='relu'),  
    layers.MaxPooling2D((2, 2)),  
    layers.Conv2D(64, (3, 3), activation='relu'), 
    layers.MaxPooling2D((2, 2)),  
    layers.Flatten(),  
    layers.Dense(128, activation='relu'),  
    layers.Dropout(0.5),  
    layers.Dense(64, activation='relu'),  
    layers.Dense(10, activation='softmax')  
])

# %% [code] {"execution":{"iopub.status.busy":"2023-11-21T06:33:06.874231Z","iopub.execute_input":"2023-11-21T06:33:06.874547Z","iopub.status.idle":"2023-11-21T06:33:06.892471Z","shell.execute_reply.started":"2023-11-21T06:33:06.874521Z","shell.execute_reply":"2023-11-21T06:33:06.891641Z"}}
# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# %% [code] {"execution":{"iopub.status.busy":"2023-11-21T06:33:06.893643Z","iopub.execute_input":"2023-11-21T06:33:06.893950Z","iopub.status.idle":"2023-11-21T06:33:06.929768Z","shell.execute_reply.started":"2023-11-21T06:33:06.893924Z","shell.execute_reply":"2023-11-21T06:33:06.926362Z"}}
# Print model summary
model.summary()

# %% [code] {"execution":{"iopub.status.busy":"2023-11-21T06:33:06.933733Z","iopub.execute_input":"2023-11-21T06:33:06.934095Z","iopub.status.idle":"2023-11-21T06:33:43.248511Z","shell.execute_reply.started":"2023-11-21T06:33:06.934061Z","shell.execute_reply":"2023-11-21T06:33:43.247675Z"}}

# Train the model using model.fit()
model.fit(train_images, train_labels, batch_size=320, epochs=50, validation_data=(val_images, val_labels))

# %% [code] {"execution":{"iopub.status.busy":"2023-11-21T06:33:43.249491Z","iopub.execute_input":"2023-11-21T06:33:43.249757Z","iopub.status.idle":"2023-11-21T06:33:45.445814Z","shell.execute_reply.started":"2023-11-21T06:33:43.249735Z","shell.execute_reply":"2023-11-21T06:33:45.444857Z"}}
# Use this trained model to predict on test_images
prediction = model.predict(test_images)
print(prediction.shape)

# %% [code] {"execution":{"iopub.status.busy":"2023-11-21T06:33:45.447159Z","iopub.execute_input":"2023-11-21T06:33:45.447522Z","iopub.status.idle":"2023-11-21T06:33:45.550314Z","shell.execute_reply.started":"2023-11-21T06:33:45.447489Z","shell.execute_reply":"2023-11-21T06:33:45.549635Z"}}
# The shape of prediction result is (28000, num_classes), assuming num_classes is the number of categories (which is 10 here)
num_classes = prediction.shape[1]

# Create and open a CSV file
with open("C:\\Users\\versha.shukla\\Desktop\\project\\submission.csv", mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write headers to the CSV file
    writer.writerow(['ImageId', 'Label'])

   # Iterate through each prediction result
    for i, pred in enumerate(prediction):
       # Get the index of the class with the highest probability in the prediction result
        label = pred.argmax()

        # Write ImageId and Label of this prediction result to the CSV file 
        writer.writerow([i+1, label])

# %% [code]
