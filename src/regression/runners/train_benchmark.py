import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from utils import dataloader, preprocessing
from model.benchmark import LinearModel

df = dataloader.loadCrimeDataset()

scalar_features = ['LAT', 'LON', 'Vict Age']
categorical_features = ['Year OCC', 'Month OCC', 'Vict Sex']
target_feature = ['Count']
source_features = scalar_features + categorical_features
all_features = source_features + target_feature

data = df[all_features].copy()
data = data.dropna()
data['LAT'] = data['LAT'].apply(lambda x: np.round(x, 2))
data['LON'] = data['LON'].apply(lambda x: np.round(x, 2))

data = data[(data.T != 0).all()]

dataset = data.groupby(source_features).agg('sum').reset_index()
dataset = shuffle(dataset, random_state = 0)

# Encoding dataset
dataset_enc = []
feature_end_ids = []

# scalar features:
scalar_data = dataset[scalar_features].to_numpy()
scalar_mean = scalar_data.mean(axis=0)
scalar_std = scalar_data.std(axis=0)
scalar_norm = (scalar_data - scalar_mean) / scalar_std
dataset_enc.append(scalar_norm)
feature_end_ids.extend((np.arange(len(scalar_features)) + 1).tolist())

# categorial features:
categories = { c: dataset[c].unique() for c in categorical_features }

for c in categorical_features:
  dataset_enc.append(preprocessing.one_hot_encoding(categories[c], dataset[c].to_numpy()))
  if len(feature_end_ids) > 0:
    feature_end_ids.append(feature_end_ids[-1] + len(categories[c]))
  else:
    feature_end_ids.append([len(categories[c])])

# prepare train/test data
dataset_enc = np.concatenate(dataset_enc, axis=1)
print(dataset_enc.shape)

labels = dataset[target_feature].to_numpy().flatten()
print(labels.shape)

num_rows, num_features = dataset_enc.shape

random_seed = 0
torch.manual_seed(random_seed)
np.random.seed(random_seed)

test_size = 0.2

num_epoch = 100
learning_rate = .0001
batch_size = 256
decay = 0.95

x_train, x_test, y_train, y_test = train_test_split(
    dataset_enc, labels, test_size=test_size, random_state=random_seed)

print(x_train.shape)
print(x_test.shape)

# train benchmark
linear_model = LinearModel(num_features=num_features)
losses = linear_model.train(
    x_train, y_train, num_epoch=num_epoch, 
    batch_size=batch_size, learning_rate=learning_rate, decay=decay)

# draw training curve
plt.clf()
plt.plot(np.arange(len(losses)), losses)
plt.title("Training Curve")
plt.ylabel("MSE Loss")
plt.xlabel("Epoch")
plt.grid()
plt.show()

# testing
metric = linear_model.test(x_test, y_test, batch_size=batch_size)
print(f"Final metrics of linear regression model = {metric:.5f}")

metric_rela = linear_model.test_rela(x_test, y_test, batch_size=batch_size)
print(f"Final relative metrics of linear regression model = {metric_rela:.5f}")

# model comprehension

weights = linear_model.coeffs.detach().numpy()[1:,0]
ticks = scalar_features + [ c + ' - ' + str(sub_c) for c in categorical_features for sub_c in sorted(categories[c]) ]

neg_mask = weights < 0
pos_mask = weights >= 0

plt.clf()
plt.figure(figsize=(10, 10))
plt.barh(np.arange(len(ticks))[neg_mask], weights[neg_mask], align='center', color='red', label="negative related")
plt.barh(np.arange(len(ticks))[pos_mask], weights[pos_mask], align='center', color='green', label="positive related")
plt.yticks(ticks = np.arange(len(ticks)), labels = ticks)
plt.title('Visualization of Trained Model Weights')
plt.ylabel('Feature')
plt.xlabel('Weight')
plt.xlim([weights.min(), weights.max()])
plt.grid()
plt.legend()
plt.show()

