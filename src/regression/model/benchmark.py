import torch
import numpy as np

class LinearModel:
  def __init__(self, num_features):
    self.coeffs = torch.zeros([num_features + 1, 1]).to(torch.float64)
  
  def predict(self, x):
    x_tensor = torch.from_numpy(x).to(torch.float64)
    return self.coeffs[0, 0] + x_tensor @ self.coeffs[1:, :]
  
  def metrics(self, y_pred, y):
    y_tensor = torch.from_numpy(y).to(torch.float64)
    return torch.mean(torch.abs(y_tensor - y_pred)).detach().numpy()
  
  def metrics_rela(self, y_pred, y):
    y_tensor = torch.from_numpy(y).to(torch.float64)
    return torch.mean(torch.abs(y_tensor - y_pred) / (y_tensor + 1e-5)).detach().numpy()
  
  def mse_loss(self, y_pred, y):
    y_tensor = torch.from_numpy(y).to(torch.float64)
    return torch.mean((y_tensor - y_pred) ** 2)

  def train(self, x_train, y_train, num_epoch = 10, batch_size=64, learning_rate = 1e-3, decay=0.95):
    num_iter_per_epoch = x_train.shape[0] // batch_size + 1
    losses = []
    for e in range(num_epoch+1):
      losses_epoch = []

      idx = np.random.choice(np.arange(x_train.shape[0]), x_train.shape[0], replace=False)
      x_train = x_train[idx]
      y_train = y_train[idx]

      for it in range(num_iter_per_epoch):
        if it == num_iter_per_epoch - 1:
          x_train_batch = x_train[batch_size * it:]
          y_train_batch = y_train[batch_size * it:]
        else:
          x_train_batch = x_train[batch_size * it:batch_size * (1+it)]
          y_train_batch = y_train[batch_size * it:batch_size * (1+it)]

        self.coeffs.requires_grad_()
        y_pred = self.predict(x_train_batch)
        loss = self.mse_loss(y_pred, y_train_batch)

        with torch.no_grad():
            if self.coeffs.grad is not None:
                self.coeffs.grad.zero_()
            loss.backward()
            self.coeffs.sub_(learning_rate * self.coeffs.grad)

        losses_epoch.append(loss.detach().numpy())
      losses.append(np.mean(losses_epoch))
      if e % (num_epoch // 10) == 0:
        print(f"Epoch {e:02d}: loss - {losses[-1]:.5f}")
      
      learning_rate *= decay
    return losses

  def test(self, x_test, y_test, batch_size=64):
    num_iter_per_epoch = x_test.shape[0] // batch_size + 1

    metric = 0

    for it in range(num_iter_per_epoch):
      if it == num_iter_per_epoch - 1:
        x_test_batch = x_test[batch_size * it:]
        y_test_batch = y_test[batch_size * it:]
      else:
        x_test_batch = x_test[batch_size * it:batch_size * (1+it)]
        y_test_batch = y_test[batch_size * it:batch_size * (1+it)]

      with torch.no_grad():
        y_pred = self.predict(x_test_batch)
        metric_batch = self.metrics(y_pred, y_test_batch)
        metric += metric_batch * y_test_batch.shape[0]

    return metric / x_test.shape[0]
  
  def test_rela(self, x_test, y_test, batch_size=64):
    num_iter_per_epoch = x_test.shape[0] // batch_size + 1

    metric = 0

    for it in range(num_iter_per_epoch):
      if it == num_iter_per_epoch - 1:
        x_test_batch = x_test[batch_size * it:]
        y_test_batch = y_test[batch_size * it:]
      else:
        x_test_batch = x_test[batch_size * it:batch_size * (1+it)]
        y_test_batch = y_test[batch_size * it:batch_size * (1+it)]

      with torch.no_grad():
        y_pred = self.predict(x_test_batch)
        metric_batch = self.metrics_rela(y_pred, y_test_batch)
        metric += metric_batch * y_test_batch.shape[0]

    return metric / x_test.shape[0]