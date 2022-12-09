import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import dataloader, preprocessing
from model.RNN import RNNModel

df = dataloader.loadCrimeDataset()

scalar_features = ['LAT', 'LON']
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
dataset = dataset.dropna()
print(f"Dataset size: {dataset.shape}")
dataset.head(5)

# grouping data for (lat, lng, sex)
group_features_sex = ['LAT', 'LON', 'Vict Sex']
gb_sex = dataset.groupby(group_features_sex)
print(f"Total number of groups for (lat, lon, sex): {len(gb_sex.groups)}")

group_data = {}
for x in gb_sex.groups:
  data = gb_sex.get_group(x)
  if len(data) >= 5:
    group_data[x] = data
print(f"Total number of valid groups for (lat, lon, sex): {len(group_data)}")

dataset_processed = {}
# scalar features:
scalar_data = dataset[scalar_features].to_numpy()
scalar_mean = scalar_data.mean(axis=0)
scalar_std = scalar_data.std(axis=0)
# categorial features:
categories = { c: dataset[c].unique() for c in categorical_features }


for key in group_data:
  dataset_key = group_data[key]
  # Encoding dataset
  dataset_enc = []
  feature_end_ids = []
  scalar_data = dataset_key[scalar_features].to_numpy()
  scalar_norm = (scalar_data - scalar_mean) / scalar_std
  dataset_enc.append(scalar_norm)
  feature_end_ids.extend((np.arange(len(scalar_features)) + 1).tolist())

  for c in categorical_features:
    dataset_enc.append(preprocessing.one_hot_encoding(categories[c], dataset_key[c].to_numpy()))
    if len(feature_end_ids) > 0:
      feature_end_ids.append(feature_end_ids[-1] + len(categories[c]))
    else:
      feature_end_ids.append([len(categories[c])])

  dataset_enc = np.concatenate(dataset_enc, axis=1)

  labels = dataset_key[target_feature].to_numpy().flatten()

  dataset_processed[key] = {'data': dataset_enc, "label": labels}

print(len(dataset_processed))

num_rows, num_features = dataset_enc.shape

random_seed = 0
torch.manual_seed(random_seed)
np.random.seed(random_seed)

test_size = 0.2
validation_size = 0.2

num_epoch = 1000
learning_rate = .01

keys = list(dataset_processed.keys())
num_keys = len(keys)

num_test = round(num_keys * test_size)
num_val = round(num_keys * validation_size)
num_train = num_keys - num_test - num_val

print(num_train, num_val, num_test)

np.random.seed(random_seed)
np.random.shuffle(keys)


train_keys = keys[:num_train]
test_keys = keys[num_train:num_train+num_test]
val_keys = keys[-num_val:]
print(val_keys[:3])

in_dims = dataset_processed[train_keys[0]]['data'].shape[1]

# train benchmark
torch.manual_seed(random_seed)
model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rnn_model = RNNModel(input_dims=in_dims, hidden_dims=8, dropout_rate=0.0)
rnn_model.to(model_device)
losses = rnn_model.train_model(
    dataset_processed, train_keys, num_epoch=num_epoch, learning_rate=learning_rate)

plt.clf()
plt.plot(np.arange(len(losses)), losses)
plt.title("Training Curve")
plt.ylabel("MSE Loss")
plt.xlabel("Epoch")
plt.grid()
plt.show()

rnn_model.to(rnn_model.device)
metric = rnn_model.test(dataset_processed, test_keys)
print(f"Final metrics of rnn model = {metric:.5f}")

optimizer_types = {"SGD": torch.optim.SGD, "Adam": torch.optim.Adam, "RMSprop": torch.optim.RMSprop}
losses = []
metrics = []
models = []

for (optimizer_type, optimizer_class) in optimizer_types.items():
  torch.manual_seed(random_seed)
  rnn_model_c = RNNModel(input_dims=in_dims, hidden_dims=8)
  rnn_model_c.to(rnn_model_c.device)
  losses_c = rnn_model_c.train_model(
      dataset_processed, train_keys, num_epoch=num_epoch, 
      learning_rate=learning_rate, optimizer_type = optimizer_class)
  losses.append(losses_c)

  metric_c = rnn_model_c.test(dataset_processed, test_keys)
  metrics.append(metric_c)
  models.append(rnn_model_c)
np.savetxt("effect_optimizer_losses.npy", np.array(losses))
np.savetxt("effect_optimizer_metrics.npy", np.array(metrics))

# Plot the effect
losses = np.loadtxt("effect_optimizer_losses.npy")
metrics = np.loadtxt("effect_optimizer_metrics.npy")
epochs = np.arange(losses.shape[1])

print(metrics)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for (i, optimizer_type) in enumerate(optimizer_types):
  loss_data = losses[i]
  metric_data = metrics[i]
  axes[0].plot(epochs, loss_data, label=optimizer_type)

axes[0].legend()
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("MSE Loss")

xticks = np.arange(metrics.shape[0])
axes[1].bar(xticks, metrics)
axes[1].set_xticks(xticks)
axes[1].set_xticklabels(list(optimizer_types.keys()))
axes[1].set_xlabel("Optimizer Type")
axes[1].set_ylabel("Metric")
fig.suptitle("Effect of different optimizers")
plt.show()

opt_optimizer = torch.optim.RMSprop

# Hidden Layer Dimensions

hidden_layer_dims = [4, 8, 16]

metrics = []
for hidden_layer_dim in hidden_layer_dims:
  rnn_model_c = RNNModel(input_dims=in_dims, hidden_dims=hidden_layer_dim)
  rnn_model_c.to(rnn_model_c.device)
  _ = rnn_model_c.train_model(
      dataset_processed, train_keys, num_epoch=num_epoch, 
      learning_rate=learning_rate, optimizer_type = opt_optimizer)

  metric_c = rnn_model_c.test(dataset_processed, val_keys)
  metrics.append(metric_c)

  print(metric_c)

print(f"Hidden Layer Dimensions: {hidden_layer_dims}")
print(f"Metrics on validation set: {metrics}")

opt_id = np.argmin(metrics)
opt_hidden_layer_dim = hidden_layer_dims[opt_id]
print(f"Optimal hidden layer dimension: {opt_hidden_layer_dim}")

torch.manual_seed(random_seed)
opt_model = RNNModel(input_dims=in_dims, hidden_dims=opt_hidden_layer_dim)
opt_model.to(opt_model.device)
_ = opt_model.train_model(
    dataset_processed, train_keys + val_keys, num_epoch=num_epoch, 
    learning_rate=learning_rate, optimizer_type = opt_optimizer)

metric = opt_model.test(dataset_processed, test_keys)
print(f"Optimal model metric: {metric:.5f} on test set")
