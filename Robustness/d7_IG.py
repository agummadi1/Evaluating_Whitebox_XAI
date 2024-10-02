# %%
import warnings
warnings.filterwarnings('ignore') 

from adversarial_models import * 
from utils import *
from get_data import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

import lime
import lime.lime_tabular
import shap
from copy import deepcopy


# %%
params = Params("summer24/Robustness/model_configurations/experiment_params.json")
np.random.seed(params.seed)
X, y, cols = get_and_preprocess_iot7(params)

# Add a random column -- this is what we'll have LIME/SHAP explain.
X['unrelated_column'] = np.random.choice([0,1],size=X.shape[0])
features = [c for c in X]

categorical_feature_name = ['HH_L5_magnitude','HH_L1_magnitude','HH_L3_magnitude','HH_L3_mean','HH_L5_weight','unrelated_column']

categorical_feature_indcs = [features.index(c) for c in categorical_feature_name]

race_indc = features.index('HH_L5_magnitude')
unrelated_indcs = features.index('unrelated_column')
X = X.values

# %%
# race_indc

# %%
from collections import Counter
print(Counter(y))

# %%
#to simulate an attack heavily biased on one feature.
class racist_model_f:
    # Decision rule: classify negatively if race is black
    def predict(self,X):
        return np.array([params.negative_outcome if x[race_indc] > 0 else params.positive_outcome for x in X])

    def predict_proba(self, X): 
        return one_hot_encode(self.predict(X))

    def score(self, X,y):
        return np.sum(self.predict(X)==y) / len(X)

#to hide the biased feature    
class innocuous_model_psi:
    # Decision rule: classify according to randomly drawn column 'unrelated column'
    def predict(self,X):
        return np.array([params.negative_outcome if x[unrelated_indcs] > 0 else params.positive_outcome for x in X])

    def predict_proba(self, X): 
        return one_hot_encode(self.predict(X))

    def score(self, X,y):
        return np.sum(self.predict(X)==y) / len(X)

# %%
# Split the data and normalize
xtrain,xtest,ytrain,ytest = train_test_split(X,y)
xtest_not_normalized = deepcopy(xtest)
ss = StandardScaler().fit(xtrain)
xtrain = ss.transform(xtrain)
xtest = ss.transform(xtest)



# %%

import tensorflow as tf

# Now you can use Keras modules directly from tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Flatten
import innvestigate
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.callbacks import LearningRateScheduler


# %%

# Train the adversarial model
adv_shap = Adversarial_Kernel_SHAP_Model(racist_model_f(), innocuous_model_psi()).\
            train(xtrain, ytrain, feature_names=features, estimator = True)

# %%
# # Set the background distribution for the shap explainer using kmeans
# background_distribution = shap.kmeans(xtrain,10)

# # Let's use the shap kernel explainer and grab a point to explain
# to_examine = np.random.choice(xtest.shape[0])

# # # Explain the biased model
# biased_kernel_explainer = shap.KernelExplainer(racist_model_f().predict, background_distribution)
# biased_shap_values = biased_kernel_explainer.shap_values(xtest[to_examine:to_examine+1])

# # Convert DenseData to numpy array
# background_distribution_array = background_distribution.data

# # Explain the adversarial model
# adv_kerenel_explainer = shap.DeepExplainer(adv_shap.perturbation_identifier, background_distribution_array)
# adv_shap_values = adv_kerenel_explainer.shap_values(xtest[to_examine:to_examine+1])

# # Plot it using SHAP's plotting features.
# shap.summary_plot(biased_shap_values, feature_names=features, plot_type="bar")
# shap.summary_plot(adv_shap_values[0], feature_names=features, plot_type="bar")

# print ("Fidelity: {0:3.2}".format(adv_shap.fidelity(xtest[to_examine:to_examine+1])))



# %%
import matplotlib.pyplot as plt




# Set the background distribution for the shap explainer using kmeans
background_distribution = shap.kmeans(xtrain,10)

# Let's use the shap kernel explainer and grab a point to explain
to_examine = np.random.choice(xtest.shape[0])

# # Explain the biased model
biased_kernel_explainer = shap.KernelExplainer(racist_model_f().predict, background_distribution)
biased_shap_values = biased_kernel_explainer.shap_values(xtest[to_examine:to_examine+1])

# %%

import innvestigate
tf.compat.v1.disable_eager_execution()

sample = xtest[to_examine:to_examine+1]

analyzer = innvestigate.create_analyzer("integrated_gradients", adv_shap.perturbation_identifier)
analysis = analyzer.analyze(sample)

# names = xtrain.columns
names = features

scores = pd.DataFrame(analysis)
scores_abs = scores.abs()
sum_of_columns = scores_abs.sum(axis=0)

names = list(names)

sum_of_columns = list(sum_of_columns)

sum_of_columns
combined = list(zip(names, sum_of_columns))
# Sort the combined list in descending order based on the values from sum_of_columns
sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)

# Unzip the sorted_combined list to separate names and sum_of_columns
sorted_names, sorted_sum_of_columns = zip(*sorted_combined)
shap_val = sorted_sum_of_columns
feature_name = sorted_names
# sorted_names

ig_array = np.array(shap_val)
feature_name = list(feature_name)


# %%
# Plot it using SHAP's plotting features.
shap.summary_plot(biased_shap_values, feature_names=features, plot_type="bar")
# shap.summary_plot(ig_array, feature_names=feature_name, plot_type="bar")

plt.savefig('Robustness_d7_IG_1.png')
plt.clf()

# Create a list of labels (optional)
labels =  feature_name # Replace this with your actual labels

labels = [x for _, x in sorted(zip(ig_array, labels), reverse=False)]
ig_array = sorted(ig_array, reverse=False)

# Create a horizontal bar plot
plt.barh(labels, ig_array)


# Display the plot
# plt.show()


# plt.plot(thresholds, sparsity_values, marker='o', linestyle='-')
# plt.xlabel('Threshold')
# plt.ylabel('Sparsity')
# plt.title('Sparsity vs. Threshold')
plt.savefig('Robustness_d7_IG_2.png')
plt.clf()

# print ("Fidelity: {0:3.2}".format(adv_shap.fidelity(xtest[to_examine:to_examine+1])))



# %%
xtest.shape[0]

# %%
# 4 buckets, 3 graphs
dict_biased_1 = {
    'Biased': 0,
    'Unrelated_1':0,
    'Unrelated_2':0,
    'Others':0
}
dict_biased_2 = {
    'Biased': 0,
    'Unrelated_1':0,
    'Unrelated_2':0,
    'Others':0
}
dict_biased_3 = {
    'Biased': 0,
    'Unrelated_1':0,
    'Unrelated_2':0,
    'Others':0
}
attack_1 = {
    'Biased': 0,
    'Unrelated_1':0,
    'Unrelated_2':0,
    'Others':0
}

attack_2 = {
    'Biased': 0,
    'Unrelated_1':0,
    'Unrelated_2':0,
    'Others':0
}

attack_3 = {
    'Biased': 0,
    'Unrelated_1':0,
    'Unrelated_2':0,
    'Others':0
}


# %%
# features analyzed
biased_feature = 'HH_L5_magnitude'
Unrelated_1_feature = 'unrelated_column'
# Unrelated_2_feature = 'unrelated_column'
samples = 100

# %%
for i in range(0,samples):   
 # Set the background distribution for the shap explainer using kmeans
    background_distribution = shap.kmeans(xtrain,10)

    # Let's use the shap kernel explainer and grab a point to explain
    to_examine = i

    # Explain the biased model
    biased_kernel_explainer = shap.KernelExplainer(racist_model_f().predict, background_distribution)
    biased_shap_values = biased_kernel_explainer.shap_values(xtest[to_examine:to_examine+1])



    #--------------------

        
    import innvestigate
    tf.compat.v1.disable_eager_execution()

    sample = xtest[to_examine:to_examine+1]

    analyzer = innvestigate.create_analyzer("integrated_gradients", adv_shap.perturbation_identifier)
    analysis = analyzer.analyze(sample)

    # names = xtrain.columns
    names = features

    scores = pd.DataFrame(analysis)
    scores_abs = scores.abs()
    sum_of_columns = scores_abs.sum(axis=0)

    names = list(names)

    sum_of_columns = list(sum_of_columns)

    sum_of_columns
    combined = list(zip(names, sum_of_columns))
    # Sort the combined list in descending order based on the values from sum_of_columns
    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)

    # Unzip the sorted_combined list to separate names and sum_of_columns
    sorted_names, sorted_sum_of_columns = zip(*sorted_combined)
    shap_val = sorted_sum_of_columns
    feature_name = sorted_names

    ig_array = np.array(shap_val)
    feature_name = list(feature_name)

    # Create a list of labels (optional)
    labels =  feature_name # Replace this with your actual labels

    labels = [x for _, x in sorted(zip(ig_array, labels), reverse=False)]
    ig_array = sorted(ig_array, reverse=False)
    #--------------------



    biased_df = pd.DataFrame({
            'shap_values': biased_shap_values[0],
            'shap_values_abs': abs(biased_shap_values[0]),
            'features': features    
        })
    
    # index_prediction = np.argmax(adv_shap.perturbation_identifier.predict(xtest[0:1]))

    attack1_df = pd.DataFrame({
            'shap_values': ig_array,
            'shap_values_abs':abs(np.array(ig_array)),
            'features': labels    
        })
    # info = [adv_shap_values,features]

    attack1_df.sort_values(by=['shap_values_abs'], ascending=False,inplace=True)
    biased_df.sort_values(by=['shap_values_abs'], ascending=False,inplace=True)

    attack1_df = attack1_df.reset_index(drop=True)
    biased_df = biased_df.reset_index(drop=True)

    # print('Attack')
    # print('------------------------------')
    # print (attack1_df)
    # print('Biased')
    # print('------------------------------')
    # print(biased_df)


    #biased columns

    #For 1st position
    if biased_df['features'][0] == biased_feature: dict_biased_1['Biased'] = dict_biased_1['Biased'] + 1
    elif biased_df['features'][0] == Unrelated_1_feature: dict_biased_1['Unrelated_1'] = dict_biased_1['Unrelated_1'] + 1
    # elif biased_df['features'][0] == Unrelated_2_feature: dict_biased_1['Unrelated_2'] = dict_biased_1['Unrelated_2'] + 1
    else: dict_biased_1['Others'] = dict_biased_1['Others'] + 1

    #For 2st position
    if biased_df['features'][1] == biased_feature: dict_biased_2['Biased'] = dict_biased_2['Biased'] + 1
    elif biased_df['features'][1] == Unrelated_1_feature: dict_biased_2['Unrelated_1'] = dict_biased_2['Unrelated_1'] + 1
    # elif biased_df['features'][1] == Unrelated_2_feature: dict_biased_2['Unrelated_2'] = dict_biased_2['Unrelated_2'] + 1
    else: dict_biased_2['Others'] = dict_biased_2['Others'] + 1

    #For 3st position
    if biased_df['features'][2] == biased_feature: dict_biased_3['Biased'] = dict_biased_3['Biased'] + 1
    elif biased_df['features'][2] == Unrelated_1_feature: dict_biased_3['Unrelated_1'] = dict_biased_3['Unrelated_1'] + 1
    # elif biased_df['features'][2] == Unrelated_2_feature: dict_biased_3['Unrelated_2'] = dict_biased_3['Unrelated_2'] + 1
    else: dict_biased_3['Others'] = dict_biased_3['Others'] + 1


    #Attack 1 columns

    #For 1st position
    if attack1_df['features'][0] == biased_feature: attack_1['Biased'] = attack_1['Biased'] + 1
    elif attack1_df['features'][0] == Unrelated_1_feature: attack_1['Unrelated_1'] = attack_1['Unrelated_1'] + 1
    # elif attack1_df['features'][0] == Unrelated_2_feature: attack_1['Unrelated_2'] = attack_1['Unrelated_2'] + 1
    else: attack_1['Others'] = attack_1['Others'] + 1

    #For 2st position
    if attack1_df['features'][1] == biased_feature: attack_2['Biased'] = attack_2['Biased'] + 1
    elif attack1_df['features'][1] == Unrelated_1_feature: attack_2['Unrelated_1'] = attack_2['Unrelated_1'] + 1
    # elif attack1_df['features'][1] == Unrelated_2_feature: attack_2['Unrelated_2'] = attack_2['Unrelated_2'] + 1
    else: attack_2['Others'] = attack_2['Others'] + 1

    #For 3st position
    if attack1_df['features'][2] == biased_feature: attack_3['Biased'] = attack_3['Biased'] + 1
    elif attack1_df['features'][2] == Unrelated_1_feature: attack_3['Unrelated_1'] = attack_3['Unrelated_1'] + 1
    # elif attack1_df['features'][2] == Unrelated_2_feature: attack_3['Unrelated_2'] = attack_3['Unrelated_2'] + 1
    else: attack_3['Others'] = attack_3['Others'] + 1


# print ("Fidelity: {0:3.2}".format(adv_shap.fidelity(xtest[to_examine:to_examine+1])))

# %%
#Normalized
keys = ['Biased','Unrelated_1','Unrelated_2','Others' ]
for x in keys:

    dict_biased_1[x] = dict_biased_1[x]/samples
    dict_biased_2[x] = dict_biased_2[x]/samples
    dict_biased_3[x] = dict_biased_3[x]/samples

    attack_1[x] = attack_1[x]/samples
    attack_2[x] = attack_2[x]/samples
    attack_3[x] = attack_3[x]/samples

# %%
print(dict_biased_1 )
print(dict_biased_2 )
print(dict_biased_3 )
print('--------------')
print(attack_1 )
print(attack_2 )
print(attack_3 )

# %%
# importing package
import matplotlib.pyplot as plt
import numpy as np
total = samples 
# create data
x = ['1st', '2nd', '3rd']
y1 = np.array([dict_biased_1['Biased'], dict_biased_2['Biased'], dict_biased_3['Biased']])
y2 = np.array([dict_biased_1['Unrelated_1'], dict_biased_2['Unrelated_1'], dict_biased_3['Unrelated_1']])
y3 = np.array([dict_biased_1['Others'], dict_biased_2['Others'], dict_biased_3['Others']])
y4 = np.array([dict_biased_1['Unrelated_2'], dict_biased_2['Unrelated_2'],dict_biased_3['Unrelated_2']])


# plot bars in stack manner
plt.bar(x, y1, color='r')
plt.bar(x, y2, bottom=y1, color='b')
plt.bar(x, y3, bottom=y1+y2, color='grey')
# plt.bar(x, y4, bottom=y1+y2+y3, color='g')
plt.xlabel("Features Importance Rank", fontsize = 18)
plt.ylabel("% Ocurrence", fontsize = 18)
# plt.legend(["Biased", "Unrelated", "Unrelated_2", "Others"])
plt.legend(["Biased", "Unrelated", "Others"], fontsize = 14)

#plt.show()

plt.savefig('Robustness_d7_IG_3.png')
plt.clf()


# %%
# importing package
import matplotlib.pyplot as plt
import numpy as np
total = samples 
# create data
x = ['1st', '2nd', '3rd']
y1 = np.array([attack_1['Biased'], attack_2['Biased'], attack_3['Biased']])
y2 = np.array([attack_1['Unrelated_1'], attack_2['Unrelated_1'], attack_3['Unrelated_1']])
y3 = np.array([attack_1['Others'], attack_2['Others'],attack_3['Others']])
# y4 = np.array([attack_1['Unrelated_2'], attack_2['Unrelated_2'], attack_3['Unrelated_2']])

# plot bars in stack manner
plt.bar(x, y1, color='r')
plt.bar(x, y2, bottom=y1, color='b')
plt.bar(x, y3, bottom=y1+y2, color='grey')
# plt.bar(x, y4, bottom=y1+y2+y3, color='g')
plt.xlabel("Features Importance Rank", fontsize = 18)
plt.ylabel("% Ocurrence", fontsize = 18)
# plt.legend(["Biased", "Unrelated_1", "Unrelated_2", "Others"])
plt.legend(["Biased", "Unrelated_1", "Others"], fontsize = 14)

#plt.show()

plt.savefig('Robustness_d7_IG_4.png')
plt.clf()



