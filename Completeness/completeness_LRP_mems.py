import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
import tensorflow_addons as tfa
import time
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
import innvestigate
from tensorflow.keras.optimizers import RMSprop


# Disable eager execution
tf.compat.v1.disable_eager_execution()

# Define oversampling function
def oversample(X, y):
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    return X_resampled, y_resampled

# Load dataset | uncomment according to dataset
# MEMS
data = pd.read_csv("mems_dataset.csv")
df_max_scaled = data
data.pop('time')
y = data.pop('label')

# IoT d7
# data = pd.read_csv("device7_top_20_features.csv")
# df_max_scaled = data
# y = data.pop('label')

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

# Rename labels for better readability | uncomment according to dataset
# MEMS
label_map = {1: 'Normal', 2: 'Near-failure', 3: 'Failure'}
# IoT d7
#label_map = {1: 'benign', 2: 'gafgyt.combo', 3: 'gafgyt.junk', 4: 'gafgyt.scan', 5: 'gafgyt.tcp', 6: 'gafgyt.udp'}

data['label'] = data['label'].map(label_map)

# Separate features and labels | uncomment according to dataset
# MEMS
X = data[['x', 'y', 'z']]
# IoT d7
#X = data.drop(columns=['label'])

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
output_file_name = "Completeness_LRP_mems.txt" # mems rename

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
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # model.compile(optimizer=RMSprop(learning_rate=learning_rate),
    #           loss='categorical_crossentropy',
    #           metrics=['accuracy'])
    return model

model = build_model(input_dim, num_classes)

# Model training
print('Training the model')
start = time.time()
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
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
# MEMS
roc_auc = roc_auc_score(y_test, y_pred, average='macro')
# IOT d7
#roc_auc = roc_auc_score(label_binarize(y_test_labels, classes=np.arange(num_classes)),
                       #label_binarize(y_pred_labels, classes=np.arange(num_classes)), average='macro')

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

# filters

# MEMS
normal_samples = data[data['label'] == 'Normal']
# near_failure_samples = data[data['label'] == 'Near-failure']
# failure_samples = data[data['label'] == 'Failure']

normal_y = normal_samples.pop('label')
# near_failure_y = near_failure_samples.pop('label')
# failure_y = failure_samples.pop('label')

# d7
# benign_samples = data[data['label'] == 'benign']
# gafgyt_combo_samples = data[data['label'] == 'gafgyt.combo']
# gafgyt_junk_samples = data[data['label'] == 'gafgyt.junk']
# gafgyt_scan_samples = data[data['label'] == 'gafgyt.scan']
# gafgyt_tcp_samples = data[data['label'] == 'gafgyt.tcp']
# gafgyt_udp_samples = data[data['label'] == 'gafgyt.udp']

# benign_y = benign_samples.pop('label')
# gafgyt_combo_y = gafgyt_combo_samples.pop('label')
# gafgyt_junk_y = gafgyt_junk_samples.pop('label')
# gafgyt_scan_y = gafgyt_scan_samples.pop('label')
# gafgyt_tcp_y = gafgyt_tcp_samples.pop('label')
# gafgyt_udp_y = gafgyt_udp_samples.pop('label')

#------Cell 2 ends-----

x_axis = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]

label = ['Normal', 'Near-failure', 'Failure']
#label = ['benign', 'gafgyt.combo', 'gafgyt.junk', 'gafgyt.scan', 'gafgyt.tcp', 'gafgyt.udp']

#Define function to test sample with the waterfall plot
def waterfall_explanator(sample):

    index = np.argmax(model.predict(sample)) # Prediction of the sample
    prediction = index

    analyzer = innvestigate.create_analyzer("lrp.z", model)
    analysis = analyzer.analyze(sample)
    names = sample.columns
    scores = pd.DataFrame(analysis)
    scores_abs = scores.abs()

    sum_of_columns = scores_abs.sum(axis=0)

    names = list(names)
    sum_of_columns = list(sum_of_columns)

    sum_of_columns
    # Zip the two lists together
    combined = list(zip(names, sum_of_columns))
    # Sort the combined list in descending order based on the values from sum_of_columns
    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
    # Unzip the sorted_combined list to separate names and sum_of_columns
    sorted_names, sorted_sum_of_columns = zip(*sorted_combined)

    shap_val = sorted_sum_of_columns
    feature_name = sorted_names
    # sorted_names
    feature_val = []
    for j in sorted_names:
            feature_val.append(float(sample[j]))
    return (prediction, shap_val,feature_val,feature_name)

def completeness_all(single_class_samples, number_samples, number_of_features_pertubation):
    Bucket = {
        '0.0': 0,
        '0.1': 0,
        '0.2': 0,
        '0.3': 0,
        '0.4': 0,
        '0.5': 0,
        '0.6': 0,
        '0.7': 0,
        '0.8': 0,
        '0.9': 0,
        '1.0': 0,
    }

    Counter_all_samples = 0
    counter_samples_changed_class = 0
    print('------------------------------------------------')
    print('Initiating Completeness Experiment')
    print('------------------------------------------------')
    for i in range(0, number_samples):
        try:
            sample = single_class_samples[i:i+1]
        except:
            break # break if there more samples requested than samples in the dataset

        u = waterfall_explanator(sample)
        top_k_features = []
        top_k_features.append(u[3][0]) # append first feature
        break_condition = False
        for k in range(1, number_of_features_pertubation):
            for j in range(11):  # 11 steps to include 1.0 (0 to 10)
                if break_condition:
                    break
                perturbation = j / 10.0  # Divide by 10 to get steps of 0.1
                temp_var = sample[top_k_features[k-1]].values[0]
                result = (temp_var - perturbation) < 0
                if result:
                    sample[top_k_features[k-1]] = 1 - perturbation
                else:
                    sample[top_k_features[k-1]] = temp_var - perturbation

                v = waterfall_explanator(sample)
                if v[0] != u[0]:
                    Bucket[str(perturbation)] += 1
                    break_condition = True
                    counter_samples_changed_class += 1
                    break
                else:
                    sample[top_k_features[k-1]] = abs(temp_var - 1) # set the sample feature value as the symetric opposite
            top_k_features.append(u[3][k]) # append next feature
            if break_condition:
                break
        Counter_all_samples += 1
        progress = 100 * Counter_all_samples / number_samples
        if progress % 10 == 0:
            print('Progress', progress, '%')

    dict = Bucket
    temp = 0
    for k in dict:
        dict[k] = dict[k] + temp
        temp = dict[k]
    total = number_samples
    y_axis = []
    for k in dict:
        dict[k] = abs(dict[k] - total)
        y_axis.append(dict[k] / total)
    return (counter_samples_changed_class, Counter_all_samples, y_axis)

#------Cell 4 ends-----

K_samples = 50
# K_feat = 3 # MEMS
K_feat = 3 # d7

#------Cell 5 ends-----

#K_feat

#------Cell 6 ends-----

num_samples = K_samples
num_feat_pertubation = K_feat

#------Cell 7 ends-----

p = completeness_all(normal_samples,num_samples,num_feat_pertubation)
percentage = 100*p[0]/p[1]
with open(output_file_name, "a") as f:
     print('y_axis_normal: ', p[2], file=f)
     print(p, file=f)
     print('Number of normal samples that changed classification: ',p[0], file=f)
     print('Number of all samples analyzed: ',p[1], file=f)
     print(percentage,'%','- samples are complete ',file=f)
