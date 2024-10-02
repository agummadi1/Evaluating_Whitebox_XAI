import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import time

# Start timing
start_time = time.time()

# Combine feature columns with the label column
req_cols = ['y','label']
num_columns = 1  # 20 features
num_labels = 3  # 6 labels

fraction = 0.5  # how much of that database you want to use
frac_normal = 0.2  # how much of the normal classification you want to reduce
split = 0.70  # how you want to split the train/test data (this is percentage for train)

# Model Parameters
batch_size = 32
epochs = 10

# XAI Samples
samples = 1000

# Specify the name of the output text file
output_file_name = "mems_DNN_IntegratedGradients_output.txt"

print('--------------------------------------------------')
print('DNN')
print('--------------------------------------------------')
print('Importing Libraries')
print('--------------------------------------------------')

# Load your dataset
df = pd.read_csv('mems_dataset.csv', usecols=req_cols)

# Separate features (X) and labels (y)
X = df.drop(columns=['label'])
y = df['label']

# Map labels from 1 to 6 to 0 to 5
y -= 1

# Train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split, random_state=42)

# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(num_columns,)),
    Dense(64, activation='relu'),
    Dense(num_labels, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)

# Calculate accuracy
accuracy = model.evaluate(X_test, y_test, verbose=0)[1]

# Integrated Gradients function
@tf.function
def integrated_gradients(model, baseline, input, steps=50):
    # Scale input and compute gradients
    scaled_inputs = [baseline + (float(i) / steps) * (input - baseline) for i in range(steps + 1)]
    with tf.GradientTape() as tape:
        tape.watch(scaled_inputs)
        logits = [model(scaled_input) for scaled_input in scaled_inputs]
    grads = tape.gradient(logits, scaled_inputs)
    
    # Average gradients and compute integrated gradients
    avg_grads = tf.reduce_mean(grads, axis=0)
    integrated_grads = (input - baseline) * avg_grads
    return integrated_grads

# Baseline for Integrated Gradients
baseline = tf.zeros((1, num_columns))

# Compute Integrated Gradients for each test sample
integrated_grads = []
for i in range(len(X_test)):
    input = tf.convert_to_tensor([X_test.iloc[i].values], dtype=tf.float32)
    ig = integrated_gradients(model, baseline, input)
    integrated_grads.append(ig.numpy())

# Convert list to NumPy array for easier manipulation
integrated_grads = np.array(integrated_grads).squeeze()

# Calculate the mean absolute Integrated Gradients values for each feature
mean_abs_ig_values = np.mean(np.abs(integrated_grads), axis=0)

# Sort and print feature importance
zipped_lists = list(zip(X_train.columns.values, mean_abs_ig_values))
zipped_lists.sort(key=lambda x: x[1], reverse=True)

sorted_list1, sorted_list2 = [list(x) for x in zip(*zipped_lists)]

print('Feature Importance (Descending Order):')
for k, v in zip(sorted_list1, sorted_list2):
    print(k, v)

# Generate sparsity graph
thresholds = [i / 10 for i in range(11)]
sparsity_values = []

for threshold in thresholds:
    count_below_threshold = sum(1 for value in mean_abs_ig_values if value < threshold)
    sparsity_values.append(count_below_threshold / len(mean_abs_ig_values))

print('Sparsity:', sparsity_values)

# Save sparsity graph
plt.plot(thresholds, sparsity_values, marker='o', linestyle='-')
plt.xlabel('Threshold')
plt.ylabel('Sparsity')
plt.title('Sparsity vs. Threshold')
plt.savefig('mems_sparsity_DNN_IntegratedGradients.png')
plt.clf()

# Write results to output file
with open(output_file_name, "a") as f:
    print('\n--------------------------------------------------', file=f)
    print('DNN', file=f)
    print('--------------------------------------------------', file=f)
    print('Feature Importance (Descending Order):', file=f)
    for k, v in zip(sorted_list1, sorted_list2):
        print(k, v, file=f)
    print('Accuracy:', accuracy, file=f)
    print('Sparsity:', sparsity_values, file=f)
    print('Samples:', samples, file=f)

# End timing
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time

# Calculate execution time in hours
execution_time_hours = execution_time / 3600  # 3600 seconds in an hour

# Print execution time in hours
print("Execution time: %s hours" % execution_time_hours)

# Write execution time to output file in hours
with open(output_file_name, "a") as f:
    print('Execution time:', execution_time_hours, 'hours', file=f)
