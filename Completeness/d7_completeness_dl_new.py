import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
import time
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import shap  # Importing SHAP for Deep SHAP

# Disable eager execution
# tf.compat.v1.disable_eager_execution()

# Define oversampling function
def oversample(X, y):
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    return X_resampled, y_resampled

# Load dataset | uncomment according to dataset
data = pd.read_csv("d7_20_subset.csv")
df_max_scaled = data
y = data.pop('label')

print('---------------------------------------------------------------------------------')
print('Normalizing database')
print('---------------------------------------------------------------------------------')
print('')

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the DataFrame
scaled_df = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
data = scaled_df

data = data.assign(label=y)

# Rename labels for better readability
label_map = {1: 'benign', 2: 'gafgyt.combo', 3: 'gafgyt.junk', 4: 'gafgyt.scan', 5: 'gafgyt.tcp', 6: 'gafgyt.udp'}

data['label'] = data['label'].map(label_map)

# Separate features and labels
X = data.drop(columns=['label'])
y = data['label']

# Define DNN model parameters
input_dim = X.shape[1]
num_classes = len(label_map)
learning_rate = 0.001
epochs = 100
batch_size = 32

# Convert labels to categorical
y = pd.Categorical(y)
y = to_categorical(y.codes)

# Define XAI parameters
output_file_name = "d7_Completeness_Deep_SHAP_final.txt"  # updated for Deep SHAP

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build DNN model
def build_model(input_dim, num_classes):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model(input_dim, num_classes)

# Model training
print('Training the model')
start = time.time()
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
end = time.time()
print('ELAPSED TIME MODEL TRAINING:', (end - start) / 60, 'min')

# Model prediction
print('Predicting using the model')
start = time.time()
y_pred = model.predict(X_test)
end = time.time()
print('ELAPSED TIME MODEL PREDICTION:', (end - start) / 60, 'min')

# Convert predictions back to labels
y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

# Evaluation metrics
accuracy = accuracy_score(y_test_labels, y_pred_labels)
f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')
conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)

roc_auc = roc_auc_score(label_binarize(y_test_labels, classes=np.arange(num_classes)),
                       label_binarize(y_pred_labels, classes=np.arange(num_classes)), average='macro')

# Write outputs to file
with open(output_file_name, "a") as f:
    print('---------------------------------------------------------------------------------', file=f)
    print('Training the model', file=f)
    print('ELAPSED TIME MODEL TRAINING:', (end - start) / 60, 'min', file=f)
    print('Predicting using the model', file=f)
    print('ELAPSED TIME MODEL PREDICTION:', (end - start) / 60, 'min', file=f)
    print('Accuracy:', accuracy, file=f)
    print('F1 Score:', f1, file=f)
    print('Confusion Matrix:', file=f)
    print(conf_matrix, file=f)
    print('ROC AUC Score:', roc_auc, file=f)

#------Cell 1 ends-----


benign_samples = data[data['label'] == 'benign']
# gafgyt_combo_samples = data[data['label'] == 'gafgyt.combo']
# gafgyt_junk_samples = data[data['label'] == 'gafgyt.junk']
# gafgyt_scan_samples = data[data['label'] == 'gafgyt.scan']
# gafgyt_tcp_samples = data[data['label'] == 'gafgyt.tcp']
# gafgyt_udp_samples = data[data['label'] == 'gafgyt.udp']
# with open(output_file_name, "a") as f:
#     print('---------------------------------------------------------------------------------', file=f)
#     print("Count of benign samples:", len(benign_samples), file=f)
#     print("Count of combo samples:", len(gafgyt_combo_samples), file=f)
#     print("Count of junk samples:", len(gafgyt_junk_samples), file=f)
#     print("Count of scan samples:", len(gafgyt_scan_samples), file=f)
#     print("Count of tcp samples:", len(gafgyt_tcp_samples), file=f)
#     print("Count of udp samples:", len(gafgyt_udp_samples), file=f)

benign_y = benign_samples.pop('label')
# gafgyt_combo_y = gafgyt_combo_samples.pop('label')
# gafgyt_junk_y = gafgyt_junk_samples.pop('label')
# DONE gafgyt_scan_y = gafgyt_scan_samples.pop('label')
# gafgyt_tcp_y = gafgyt_tcp_samples.pop('label')
# gafgyt_udp_y = gafgyt_udp_samples.pop('label')

#Define function to test sample with Deep SHAP
# def waterfall_explanator(sample):

#     explainer = shap.DeepExplainer(model, X_train)  # Create Deep SHAP explainer
#     shap_values = explainer.shap_values(sample)  # Get SHAP values for the sample

#     index = np.argmax(model.predict(sample))  # Prediction of the sample
#     prediction = index

#     # Process SHAP values
#     feature_name = list(sample.columns)
#     shap_val = np.abs(shap_values[0]).sum(axis=0)

#     # Sort the SHAP values and corresponding feature names
#     combined = list(zip(feature_name, shap_val))
#     sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)

#     # Unzip the sorted list into features and SHAP values
#     sorted_names, sorted_shap_vals = zip(*sorted_combined)
    
#     feature_val = [float(sample[j]) for j in sorted_names]  # Get feature values
    
#     return (prediction, sorted_shap_vals, feature_val, sorted_names)

#Define function to test sample with Deep SHAP
def waterfall_explanator(sample):

    explainer = shap.DeepExplainer(model, X_train)  # Create Deep SHAP explainer
    
    # Convert sample DataFrame to a NumPy array
    sample_array = sample.to_numpy()
    
    # SHAP expects the input in array format
    shap_values = explainer.shap_values(sample_array)  # Get SHAP values for the sample

    # Prediction of the sample
    index = np.argmax(model.predict(sample_array)) 
    prediction = index

    # Process SHAP values
    feature_name = list(sample.columns)
    shap_val = np.abs(shap_values[0]).sum(axis=0)

    # Sort the SHAP values and corresponding feature names
    combined = list(zip(feature_name, shap_val))
    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)

    # Unzip the sorted list into features and SHAP values
    sorted_names, sorted_shap_vals = zip(*sorted_combined)
    
    # Get feature values
    feature_val = [float(sample[j]) for j in sorted_names]
    
    return (prediction, sorted_shap_vals, feature_val, sorted_names)


# Completeness function
def completeness_all(single_class_samples, number_samples, number_of_features_pertubation):
    counter_samples_changed_class = 0
    Counter_all_samples = 0
    y_axis = []  # To track the perturbation deltas

    for i in range(number_samples):
        prediction, shap_val, feature_val, feature_name = waterfall_explanator(single_class_samples.iloc[[i]])
        prediction_after_features_perturbation = []

        for j in range(number_of_features_pertubation):
            sample_modification = single_class_samples.iloc[[i]].copy()
            feature_to_zero = feature_name[j]
            sample_modification[feature_to_zero] = 0
            index = np.argmax(model.predict(sample_modification))
            prediction_after_features_perturbation.append(index)

        # Calculate the average change in prediction class
        delta = round(prediction - np.mean(prediction_after_features_perturbation), 1)
        y_axis.append(delta)

        if prediction != np.mean(prediction_after_features_perturbation):  # Classification changed
            counter_samples_changed_class += 1

        Counter_all_samples += 1

    return (counter_samples_changed_class, Counter_all_samples, y_axis)

# Set parameters
K_samples = 20
K_feat = 3  # Number of features for perturbation

# Call the function with required parameters
num_samples = K_samples
num_feat_pertubation = K_feat

result = completeness_all(benign_samples, num_samples, num_feat_pertubation)
percentage = 100 * result[0] / result[1]
# Write the result to a file
with open(output_file_name, "a") as f:
    print('---------------------------------------------------------------------------------', file=f)
    print('y_axis_benign: ', result[2], file=f)
    print('Number of benign samples that changed classification: ', result[0], file=f)
    print('Number of all samples analyzed: ', result[1], file=f)
    print(f'{percentage}% - samples are complete', file=f)

# result2 = completeness_all(gafgyt_tcp_samples, num_samples, num_feat_pertubation)
# percentage = 100 * result2[0] / result2[1]
# # Write the result to a file
# with open(output_file_name, "a") as f:
#     print('---------------------------------------------------------------------------------', file=f)
#     print('y_axis_gafgyt_tcp: ', result2[2], file=f)
#     print('Number of gafgyt_tcp samples that changed classification: ', result2[0], file=f)
#     print('Number of all samples analyzed: ', result2[1], file=f)
#     print(f'{percentage}% - samples are complete', file=f)

# result3 = completeness_all(gafgyt_udp_samples, num_samples, num_feat_pertubation)
# percentage = 100 * result3[0] / result3[1]
# # Write the result to a file
# with open(output_file_name, "a") as f:
#     print('---------------------------------------------------------------------------------', file=f)
#     print('y_axis_gafgyt_udp: ', result3[2], file=f)
#     print('Number of gafgyt_udp samples that changed classification: ', result3[0], file=f)
#     print('Number of all samples analyzed: ', result3[1], file=f)
#     print(f'{percentage}% - samples are complete', file=f)