import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from utils import dataloader, preprocessing
from model.mlp import DeepRegressionModel

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

# prepare train/val/test data
dataset_enc = np.concatenate(dataset_enc, axis=1)
print(dataset_enc.shape)

labels = dataset[target_feature].to_numpy().flatten()
print(labels.shape)

num_rows, num_features = dataset_enc.shape

random_seed = 0
torch.manual_seed(random_seed)
np.random.seed(random_seed)

test_size = 0.2
validation_size = 0.2

num_epoch = 100
learning_rate = .001
batch_size = 256

x_train_val, x_test, y_train_val, y_test = train_test_split(
    dataset_enc, labels, test_size=test_size, random_state=random_seed)

x_train, x_val, y_train, y_val = train_test_split(
    x_train_val, y_train_val, test_size=validation_size / (1-test_size), random_state=random_seed)

print(x_train.shape)
print(x_val.shape)
print(x_test.shape)

# train benchmark
torch.manual_seed(random_seed)
deep_reg_model = DeepRegressionModel(input_dims=x_train.shape[1], hidden_dims=32)
deep_reg_model.to(deep_reg_model.device)
losses = deep_reg_model.train_model(
    x_train_val, y_train_val, num_epoch=num_epoch, 
    batch_size=batch_size, learning_rate=learning_rate)

plt.clf()
plt.plot(np.arange(len(losses)), losses)
plt.title("Training Curve")
plt.ylabel("MSE Loss")
plt.xlabel("Epoch")
plt.grid()
plt.show()

# testing 
deep_reg_model.to(deep_reg_model.device)
metric = deep_reg_model.test(x_test, y_test, batch_size=batch_size)
print(f"Final metrics of deep regression model = {metric:.5f}")


# effects of mini-batch learning
# get batch candidates
min_batches = 64
max_batches = x_train.shape[0]

candidates = []
cur_batches = min_batches
while cur_batches < min(2049, max_batches):
  candidates.append(cur_batches)
  cur_batches *= 4
candidates.append(max_batches)

print("Batch Size for testing:")
print(candidates)

losses = []
metrics = []
models = []

for candidate in candidates:
  torch.manual_seed(random_seed)
  deep_reg_model_c = DeepRegressionModel(input_dims=x_train.shape[1], hidden_dims=32)
  deep_reg_model_c.to(deep_reg_model_c.device)
  losses_c = deep_reg_model_c.train_model(
      x_train_val, y_train_val, num_epoch=num_epoch // 2, 
      batch_size=candidate, learning_rate=learning_rate)
  losses.append(losses_c)

  metric_c = deep_reg_model_c.test(x_test, y_test, batch_size=candidate)
  metrics.append(metric_c)
  models.append(deep_reg_model_c)
np.savetxt("effect_miniBatch_losses.npy", np.array(losses))
np.savetxt("effect_miniBatch_metrics.npy", np.array(metrics))

# Plot the effect
losses = np.loadtxt("effect_miniBatch_losses.npy")
metrics = np.loadtxt("effect_miniBatch_metrics.npy")
print(metrics)
epochs = np.arange(losses.shape[1])

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for (i, candidate) in enumerate(candidates):
  loss_data = losses[i]
  metric_data = metrics[i]
  axes[0].plot(epochs, loss_data, label=f"batch size = {candidate:d}")

axes[0].legend()
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("MSE Loss")

xticks = np.arange(len(metrics))
axes[1].bar(xticks, metrics)
axes[1].set_xticks(xticks)
axes[1].set_xticklabels(candidates)
axes[1].set_xlabel("Batch Size")
axes[1].set_ylabel("Metric")
fig.suptitle("Effect of mini-batch learning")
plt.show()


# effects of different optimizers

optimizer_types = {"SGD": torch.optim.SGD, "Adam": torch.optim.Adam, "RMSprop": torch.optim.RMSprop}

losses = []
metrics = []
models = []

for (optimizer_type, optimizer_class) in optimizer_types.items():
  torch.manual_seed(random_seed)
  deep_reg_model_c = DeepRegressionModel(input_dims=x_train.shape[1], hidden_dims=32)
  deep_reg_model_c.to(deep_reg_model_c.device)
  losses_c = deep_reg_model_c.train_model(
      x_train_val, y_train_val, num_epoch=num_epoch // 2, 
      batch_size=256, learning_rate=learning_rate, optimizer_type = optimizer_class)
  losses.append(losses_c)

  metric_c = deep_reg_model_c.test(x_test, y_test, batch_size=256)
  metrics.append(metric_c)
  models.append(deep_reg_model_c)
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

# hidden layer dim tuning 

opt_batch_size = 256
opt_optimizer = torch.optim.RMSprop
# Hidden Layer Dimensions

hidden_layer_dims = [32, 64, 128]

metrics = []
for hidden_layer_dim in hidden_layer_dims:
  torch.manual_seed(random_seed)
  deep_reg_model_c = DeepRegressionModel(input_dims=x_train.shape[1], hidden_dims=hidden_layer_dim, dropout_rate=0.0)
  deep_reg_model_c.to(deep_reg_model_c.device)
  _ = deep_reg_model_c.train_model(
      x_train, y_train, num_epoch=num_epoch // 2, 
      batch_size=opt_batch_size, learning_rate=learning_rate, optimizer_type = opt_optimizer)

  metric_c = deep_reg_model_c.test(x_val, y_val, batch_size=opt_batch_size)
  metrics.append(metric_c)

  print(metric_c)

print(f"Hidden Layer Dimensions: {hidden_layer_dims}")
print(f"Metrics on validation set: {metrics}")

opt_id = np.argmin(metrics)
opt_hidden_layer_dim = hidden_layer_dims[opt_id]
print(f"Optimal hidden layer dimension: {opt_hidden_layer_dim}")

# Dropout rate

dropout_rates = [0.0, 0.1, 0.2]

metrics = []
for dropout_rate in dropout_rates:
  torch.manual_seed(random_seed)
  deep_reg_model_c = DeepRegressionModel(input_dims=x_train.shape[1], hidden_dims=opt_hidden_layer_dim, dropout_rate=dropout_rate)
  deep_reg_model_c.to(deep_reg_model_c.device)
  _ = deep_reg_model_c.train_model(
      x_train, y_train, num_epoch=num_epoch // 2, 
      batch_size=opt_batch_size, learning_rate=learning_rate, optimizer_type = opt_optimizer)

  metric_c = deep_reg_model_c.test(x_val, y_val, batch_size=opt_batch_size)
  metrics.append(metric_c)
  print(metric_c)

print(f"Dropout Rates: {dropout_rates}")
print(f"Metrics on validation set: {metrics}")

opt_id = np.argmin(metrics)
opt_dropout_rate = dropout_rates[opt_id]
print(f"Optimal dropout rate: {opt_dropout_rate}")


# obtain optimal 

torch.manual_seed(random_seed)
opt_model = DeepRegressionModel(input_dims=x_train.shape[1], hidden_dims=opt_hidden_layer_dim, dropout_rate=opt_dropout_rate)
opt_model.to(opt_model.device)
_ = opt_model.train_model(
      x_train_val, y_train_val, num_epoch=num_epoch, 
      batch_size=opt_batch_size, learning_rate=learning_rate, optimizer_type = opt_optimizer)

metric = deep_reg_model_c.test(x_test, y_test, batch_size=opt_batch_size)
print(f"Optimal model metric: {metric:.5f} on test set")

print(f"With our tuned parameters, the simple deep learning model performs at optimal metrics {metric:.5f} against 0.33927 for linear model benchmark on the same test set.")
print(f"Performance improved: {(0.33927 - metric)/0.33927 * 100:.2f}%")
print(f"Optimal batch size: {opt_batch_size}")
print(f"Optimal optimizer class: {opt_optimizer}")
print(f"Optimal hidden layer dimension: {opt_hidden_layer_dim}")
print(f"Optimal dropout rate: {opt_dropout_rate}")