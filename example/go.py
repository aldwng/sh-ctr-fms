import torch
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from core.models.fm import FMModel, FMwModel
from core.models.lr import LRModel

EMBED_DIM = 16

def get_dataset(name, path):
    if name == 'ctr':
        return CtrDataset(path)
    elif name == 'ctr_w':
        return CtrWDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)

def get_model(name, dataset, dataset_w):
    field_dims = dataset.field_dims
    field_dims_w = dataset.field_dims
    if name == 'lr':
        return LRModel(field_dims)
    elif name == 'fm':
        return FMModel(field_dims, EMBED_DIM)
    elif name == 'fmw':
        return FMwModel(field_dims, EMBED_DIM, field_dims_w)
    else:
        raise ValueError('unknown model name: ', name)
    
class EarlyStop(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trail_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trail_counter = 0
            torch.save(model, self.save_path)
            return True
        elif self.trail_counter + 1 < self.num_trials:
            self.trail_counter += 1
            return True
        else:
            return True
        
def fit(model, optimizer, data_loader, loss_func, device, log_interval=100):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = loss_func(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.ster()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0

def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
        return roc_auc_score(targets, predicts)
    
def go(dataset_name,
         dataset_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir):
    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)
    model = get_model(model_name, dataset).to(device)
    loss_func = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStop(num_trials=2, save_path=f'{save_dir}/{model_name}.pt')
    for epoch_i in range(epoch):
        fit(model, optimizer, train_data_loader, loss_func, device)
        auc = test(model, valid_data_loader, device)
        print('epoch:', epoch_i, 'validation: auc:', auc)
        if not early_stopper.is_continuable(model, auc):
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break
    auc = test(model, test_data_loader, device)
    print(f'test auc: {auc}')
