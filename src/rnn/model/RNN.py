import torch
import numpy as np

class RNNModel(torch.nn.Module):
  def __init__(self, input_dims, hidden_dims, gru_layers = 2, dropout_rate = 0.3, device = None):
    super(RNNModel, self).__init__()
    # encoder
    self.fc1 = torch.nn.Linear(input_dims, hidden_dims)
    # rnn
    self.gru = torch.nn.GRU(hidden_dims, hidden_dims, num_layers=gru_layers, dropout = dropout_rate)
    # decoder
    self.fc2 = torch.nn.Linear(hidden_dims, hidden_dims)
    self.fc3 = torch.nn.Linear(hidden_dims, 1)

    self.loss_fn = torch.nn.MSELoss()
    self.opt = None

    self.gru_layers = gru_layers
    self.hidden_dims = hidden_dims

    self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  def forward(self, x, h):
    x = self.fc1(x)
    x = torch.sigmoid(x)

    x, h_next = self.gru(x, h)

    x = self.fc2(x)
    x = torch.tanh(x)

    x = self.fc3(x).squeeze(dim=-1)
    x = torch.relu(x)
    return x, h_next
  
  def metrics(self, y_pred, y):
    return torch.mean(torch.abs(y - y_pred)).cpu().detach().numpy()

  def prepare_data_into_batches(self, dataset, keys):
    out_data = {}
    for key in keys:
      data = dataset[key]["data"]
      label = dataset[key]["label"]
      length = data.shape[0]
      feature_num = data.shape[1]
      if length not in out_data:
        out_data[length] = {"data": [], "label":[]}
      out_data[length]["data"].append(torch.Tensor(data).to(torch.float32).to(self.device).view(length, 1, feature_num))
      out_data[length]["label"].append(torch.Tensor(label).to(torch.float32).to(self.device).view(length, 1))
    out_x = []
    out_y = []
    for length in out_data:
      out_x.append(torch.cat(out_data[length]['data'], dim=1))
      out_y.append(torch.cat(out_data[length]['label'], dim=1))
    
    return out_x, out_y
    
  def train_model(self, dataset, keys, num_epoch = 10, learning_rate = 1e-3, optimizer_type=torch.optim.SGD):
    if self.opt is None:
      self.opt = optimizer_type(self.parameters(), lr=learning_rate)

    tensor_x, tensor_y = self.prepare_data_into_batches(dataset, keys)

    losses = []
    self.train()
    
    for e in range(num_epoch+1):
      losses_epoch = []
      for (x, y) in zip(tensor_x, tensor_y):
        h0 = torch.zeros([self.gru_layers, x.shape[1], self.hidden_dims]).to(torch.float32).to(self.device)

        pred, h_n = self.forward(x, h0)
        loss = self.loss_fn(pred, y)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        losses_epoch.append(loss.cpu().detach().numpy())
      losses.append(np.mean(losses_epoch))
      if e % (num_epoch // 10) == 0:
        print(f"Epoch {e:02d}: loss - {losses[-1]:.5f}")
      
    return losses

  def test(self, dataset, keys):
    tensor_x, tensor_y = self.prepare_data_into_batches(dataset, keys)

    metric = 0
    num_data = 0
    self.eval()
    
    with torch.no_grad():
      losses_epoch = []
      for (x, y) in zip(tensor_x, tensor_y):
        h0 = torch.zeros([self.gru_layers, x.shape[1], self.hidden_dims]).to(torch.float32).to(self.device)

        pred, h_n = self.forward(x, h0)

        metric_batch = self.metrics(pred, y)
        metric += metric_batch * pred.shape[0] * pred.shape[1]
        num_data += pred.shape[0] * pred.shape[1] 

    return metric / num_data