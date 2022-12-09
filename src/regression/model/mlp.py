import torch 
import numpy as np

class DeepRegressionModel(torch.nn.Module):
  def __init__(self, input_dims, hidden_dims, dropout_rate = 0.3, device = None):
    super(DeepRegressionModel, self).__init__()
    self.fc1 = torch.nn.Linear(input_dims, hidden_dims)
    self.fc2 = torch.nn.Linear(hidden_dims, hidden_dims//2)
    self.fc3 = torch.nn.Linear(hidden_dims//2, 1)

    self.bn1 = torch.nn.BatchNorm1d(hidden_dims)
    self.bn2 = torch.nn.BatchNorm1d(hidden_dims//2)

    self.dropout2 = torch.nn.Dropout(p=dropout_rate)
    self.dropout3 = torch.nn.Dropout(p=dropout_rate)

    self.loss_fn = torch.nn.MSELoss()
    self.opt = None

    self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  def forward(self, x):
    x = self.fc1(x)
    x = torch.sigmoid(x)
    x = self.bn1(x)
    x = self.dropout2(x,)

    x = self.fc2(x)
    x = torch.tanh(x)
    x = self.bn2(x)
    x = self.dropout3(x)

    x = self.fc3(x)
    x = torch.relu(x)
    return x
  
  def metrics(self, y_pred, y):
    return torch.mean(torch.abs(y - y_pred)).cpu().detach().numpy()

  def train_model(self, x_train, y_train, num_epoch = 10, batch_size=64, learning_rate = 1e-3, optimizer_type=torch.optim.SGD):
    if self.opt is None:
      self.opt = optimizer_type(self.parameters(), lr=learning_rate)

    tensor_x = torch.Tensor(x_train).to(torch.float32).to(self.device)
    tensor_y = torch.Tensor(y_train.reshape(-1, 1)).to(torch.float32).to(self.device)

    train_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_size = len(train_dataloader.dataset)

    num_iter_per_epoch = x_train.shape[0] // batch_size + 1
    losses = []
    self.train()
    for e in range(num_epoch+1):
      losses_epoch = []

      for batch, (X, y) in enumerate(train_dataloader):

        pred = self.forward(X)
        loss = self.loss_fn(pred, y)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        losses_epoch.append(loss.cpu().detach().numpy())
      losses.append(np.mean(losses_epoch))
      if e % (num_epoch // 10) == 0:
        print(f"Epoch {e:02d}: loss - {losses[-1]:.5f}")
      
    return losses

  def test(self, x_test, y_test, batch_size=64):
    tensor_x = torch.Tensor(x_test).to(self.device)
    tensor_y = torch.Tensor(y_test.reshape(-1, 1)).to(self.device)

    test_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    test_size = len(test_dataloader.dataset)

    metric = 0
    self.eval()

    with torch.no_grad():
      for X, y in test_dataloader:
        y_pred = self.forward(X)
        metric_batch = self.metrics(y_pred, y)
        metric += metric_batch * y_pred.shape[0]

    return metric / test_size