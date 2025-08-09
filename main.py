from utils import *
from models import *
import matplotlib.pyplot as plt

#Dataset preparation

dataset_path = "ratings.dat"
dataset = MovieLens1MDataset(dataset_path)
user_num = dataset.field_dims[0]
item_num = dataset.field_dims[1]
columns_name=['user_id','item_id','rating', 'timestamp']
df = pd.DataFrame(dataset.data, columns = columns_name)

user_num = len(df['user_id'].unique())
item_num = len(df['item_id'].unique())
print("Number of users: ", user_num, ", Number of items: ", item_num)
item_counts = df['item_id'].value_counts()
sorted_items = item_counts.sort_values(ascending=False)

item_positions = {item_id: position + 1 for position, item_id in enumerate(sorted_items.index)}

for i in range(0, item_num):
    if i not in item_positions:
        item_positions[i] = item_num

for i in range(1, item_num + 1):
    if i not in item_counts:
        item_counts[i] = 0 
ord_i = []
for i in item_counts:
    ord_i += [i - 1]

train_df = pd.read_csv("train_ml1m.csv")
val_df = pd.read_csv("val_ml1m.csv")
test_df = pd.read_csv("test_ml1m.csv")
train_data = train_df.to_numpy()[:, :4]
val_data = val_df.to_numpy()[:, :4]
test_data = test_df.to_numpy()[:, :4]

train_data = Dataset_maker(train_data)
val_data = Dataset_maker(val_data)
test_data = Dataset_maker(test_data)


# Training


class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            #torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False

        
def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    """
    Traning process
    """
    print("Training...")
    pos_mas = []
    neg_mas = []
    it_mas = []
    model.train()
    total_loss = 0
    tk0 = tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        fields_neg = fields.clone()
        fields_items = fields.clone()
        n = fields.shape[0]
        fields_neg[:, 1] = torch.randint(1, item_num, (n,))
        fields_items[:, 0] = fields_neg[:, 1].clone()
        fields_neg = fields_neg.to(device)
        fields_items = fields_items.to(device)
        y_pos = model(fields)
        y_neg = model(fields_neg)
        y_it = model(fields_items)
        loss = bpr_loss(y_pos, y_neg) + 0 * torch.linalg.norm(model.embedding.embedding.weight)
        pos_mas += [torch.linalg.norm(y_pos).item()]
        neg_mas += [torch.linalg.norm(y_neg).item()]
        it_mas += [torch.linalg.norm(y_it).item()]
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0
    return pos_mas, neg_mas, it_mas
            
            
def train2(model, optimizer, data_loader, criterion, device, log_interval=100, loss_type = "triangle"):
    """
    Traning process
    """
    print("Training...")
    pos_mas = []
    neg_mas = []
    it_mas = []
    model.train()
    total_loss = 0
    tk0 = tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        fields_neg = fields.clone()
        fields_items = fields.clone()
        n = fields.shape[0]
        #fields_neg[:, 1] = torch.randint(1, item_num, (n,))
        fields_neg[:, 1] = sampler_neg(model.embedding.embedding.weight[fields[:, 0]], model.embedding.embedding.weight[user_num:], n)
        fields_items[:, 0] = fields_neg[:, 1].clone()
        fields_neg = fields_neg.to(device)
        y1, y_pos, emb_pos = model(fields)
        y2, y_neg, emb_neg = model(fields_neg)
        y3, y_it, emb_it = model(fields_items, items = True)
        a = model.a
        b = model.b
        c = model.c
        d = model.d
        if loss_type == "bpr":
            loss = bpr_loss((a *  y1 + b), a * y2 + b)
        elif loss_type == "triangle":
            loss = triangle_loss(a * y1, (a * y2), c * y3 + d) + 0.0001 * torch.linalg.norm(y_it) ** 2
        else:
            loss = tetrahedron_loss(y_pos, y_neg, y_it, emb_pos, emb_neg, emb_it, a, b, c, d) + 0.001 * torch.linalg.norm(y_it) ** 2 
        pos_mas += [torch.linalg.norm(y_pos - emb_pos[:, 0] *  emb_pos[:, 1]).item()]
        neg_mas += [torch.linalg.norm(y_neg- emb_neg[:, 0] *  emb_neg[:, 1]).item()]
        it_mas += [torch.linalg.norm(y_it - emb_it[:, 0] *  emb_it[:, 1]).item()]
        pos_mas += [torch.linalg.norm(y_pos).item()]
        neg_mas += [torch.linalg.norm(y_neg).item()]
        it_mas += [torch.linalg.norm(y_it).item()]
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0
    return pos_mas, neg_mas, it_mas


def test(model, data_loader, device, num_users, num_items):
    #print("Testing...")
    model.eval()
    targets, predicts = list(), list()
    data = list()
    strat_time = time.time()
    with torch.no_grad():
        for fields, target in tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)

            y = torch.sigmoid(model(fields)[0].squeeze(1))

            targets.extend(target.tolist())
            predicts.extend(y.tolist())
            data.extend(fields.tolist())
    end_time = time.time()
    
    print("Convertation:")
    predicts, targets = convert(data, targets, predicts)
    max_indices = {key: np.argmax(value) for key, value in predicts.items()}
    max_inds = []
    metr = metrics(targets, predicts)
    return end_time - strat_time, 0, 0, metr["ndcg@1"],metr["ndcg@5"], metr["ndcg@10"], metr["hits@1"], metr["hits@5"],metr["hits@10"], metr["cov"]

def negative_sampling(dataset, r = 1):
    """
    Negative sampler for traning
    """
    n = dataset[:][0].shape[0]
    nb_user = int(max(dataset[:][0][:,0]))
    nb_item = int(max(dataset[:][0][:,1]))
    mas = []
    rating = []
    for i in tqdm(range(n)):
        
        mas += [[dataset[:][0][i][0], dataset[:][0][i][1]]]
        rating += [1]
    ind = 0
    k = n // 10
    for i in tqdm(range(r * n)):
        x, y = np.random.randint(0, nb_user), np.random.randint(0, nb_item)
        mas += [[x, y]]
        rating += [0]
        ind += 1
    mas = np.array(mas)
    rating = np.array(rating)
    dataset.items = mas
    dataset.targets = rating
    return dataset

def test_all_sampling(dataset, n_items, n_users):
    """
    Sample all negative examples for computing metric
    """
    n = dataset[:][0].shape[0]
    user_set = set(dataset[:][0][:,0])
    mas = []
    rating = []
    matr = np.zeros((n_users, n_items))
    for i in tqdm(range(n)):
        matr[dataset[:][0][i][0], dataset[:][0][i][1]] = 1
    for u in tqdm(user_set):
        for i in range(n_items):
            mas += [[u, i]]
            rating += [matr[u, i]]
    mas = np.array(mas)
    rating = np.array(rating)
    dataset.items = mas
    dataset.targets = rating
    return dataset



def get_model(name, dataset, embed_dim):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    field_dims = dataset.field_dims
    print(field_dims)
    if name == 'fm':
        return FactorizationMachineModel(field_dims, embed_dim=embed_dim)
    elif name == 'lfm':
        return LorentzFactorizationMachineModel(field_dims, embed_dim=embed_dim)
    elif name == 'tfm':
        return TetrahedronFactorizationMachineModel(field_dims, embed_dim=embed_dim)
    elif name == 'hyp':
        return HyperBPRModel(field_dims, embed_dim=embed_dim)
    else:
        print("No such name!")

def plot_embed(E_u, E_i, ord_u, ord_i):
    """
    Embedding visualisation
    """
    mas_u = []
    mas_i = []
    for u in ord_u:
        mas_u += [torch.linalg.norm(E_u[u]).item()]
    
    for i in ord_i:
        mas_i += [torch.linalg.norm(E_i[i]).item()]
    


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


    ax1.plot(mas_u, color='blue', label='users')
    ax1.set_xlabel('user')
    ax1.set_ylabel('norm')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid()

    ax2.plot(mas_i, color='red', label='items')
    ax2.set_xlabel('item')
    ax2.set_ylabel('norm')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.show()
    
import matplotlib.pyplot as plt
import numpy as np

def distance_count(E, tresholds):
    """
    Computing distance between two points in the hyperbolic space
    """
    mas = [0] * len(tresholds)
    for i in tqdm(range(E.shape[0])):
        d = torch.linalg.norm(E[i])
        for j in range(len(tresholds)):
            t = tresholds[j]
            if t >= d:
                mas[j] += 1
                break


    x_indices = np.arange(len(mas))
    plt.figure(figsize=(10, 6))
    plt.bar(x_indices, mas, alpha=0.7, edgecolor='black')
    plt.title('Bar Chart Example', fontsize=20)
    plt.xlabel('Names', fontsize=15)
    plt.ylabel('Values', fontsize=15)
    plt.xticks(x_indices, tresholds)
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('hypbpr_emb_diagram_16.pdf')
    plt.show()
    
def experiment2(dataset, 
         train_data,
         val_data,
         test_data,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir,
         embed_dim):
    '''
    The main experimental setup.
    '''
    print("Dataset/Model preparing...")
    av_time = []
    nb_users = user_num
    nb_items = item_num
    #test_data = test_all_sampling(test_data, nb_items)
    
    train_data_loader = DataLoader(train_data, batch_size=batch_size, num_workers=8)
    val_data_loader = DataLoader(val_data, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, num_workers=8)
    model = get_model(model_name, dataset, embed_dim)
    model.to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopper(num_trials=25, save_path=f'{save_dir}/{model_name}.pt')
    print("Training...")
    tot_pos = []
    tot_neg = []
    tot_it = []
    
    metrics = []
    
    for epoch_i in range(epoch):
        print(f"Epoch {epoch_i}")
        pos_mas, neg_mas, it_mas = train2(model, optimizer, train_data_loader, criterion, device)
        #pos_mas, neg_mas, it_mas = train(model, optimizer, train_data_loader, criterion, device)
        print(model.a.item(), model.b.item(), model.c.item(), model.d.item())
        if (epoch_i + 1) % 100 == 0:
            start_time = time.time()
            t, auc, logloss, ndcg1, ndcg5, ndcg10, hit_score1, hit_score5, hit_score10, cov = test(model, test_data_loader, device, nb_users, nb_items)
            end_time = time.time()
            av_time += [t]
            print(f'time: {t}')
            print(f'test NDCG@1: {ndcg1}')
            print(f'test NDCG@5: {ndcg5}')
            print(f'test NDCG@10: {ndcg10}')
            print(f'test Hits@1: {hit_score1}')
            print(f'test Hits@5: {hit_score5}')
            print(f'test Hits@10: {hit_score10}')
            print(f'test Cov: {cov}')
            if not early_stopper.is_continuable(model, auc):
                print(f'validation: best auc: {early_stopper.best_accuracy}')
                break
            metrics += [hit_score10]
        tot_pos += pos_mas
        tot_neg += neg_mas
        tot_it += it_mas

    plt.figure(figsize=(10, 6))
    
    np.save("pos_bpr.npy", tot_pos)
    np.save("neg_bpr.npy", tot_neg)
    np.save("it_bpr.npy", tot_it)

    plt.plot(tot_pos, label='positive', color='b')  # Синус
    plt.plot(tot_neg, label='negative', color='r')  # Косинус
    plt.plot(tot_it, label='item', color='g')  # Тангенс


    plt.xlabel('iteration')
    plt.ylabel('norm')
    plt.grid()
    plt.legend()
    plt.show()
    
    np.save("tetrafm.npy", metrics)
    
    
    tresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1]
    plot_embed(model.embedding.embedding.weight[:user_num], model.embedding.embedding.weight[user_num:], ord_u, ord_i)
    distance_count(model.embedding.embedding.weight[user_num:], tresholds)
    
    print("Testing...")
    start_time = time.time()
    t, auc, logloss, ndcg1, ndcg5, ndcg10, hit_score1, hit_score5, hit_score10, cov = test(model, test_data_loader, device, nb_users, nb_items)
    end_time = time.time()
    av_time += [t]
    print(f'time: {t}')
    print(f'test NDCG@1: {ndcg1}')
    print(f'test NDCG@5: {ndcg5}')
    print(f'test NDCG@10: {ndcg10}')
    print(f'test Hits@1: {hit_score1}')
    print(f'test Hits@5: {hit_score5}')
    print(f'test Hits@10: {hit_score10}')
    print(f'test Cov: {cov}')
    
    print(f'Average time: {np.mean(av_time)}')
    return model, mrr_score, mrr_score10, hit_score1, hit_score3, hit_score10


# Start training and hyperparameters

class Argument:

    def __init__(self):
        self.model_name = 'tfm'
        self.epoch = 20
        self.learning_rate = 0.001
        self.batch_size = 2048
        self.weight_decay = 1e-6
        self.device = 'cuda'
        self.save_dir = 'chkpt'
        self.embed_dim = 64

train_np = train_df.to_numpy()
val_np = val_df.to_numpy()
test_np = test_df.to_numpy()
train_data = Dataset_maker(train_np)
val_data = Dataset_maker(val_np)
test_data = Dataset_maker(test_np)
dataset = Dataset_maker(df.to_numpy())
user_num = dataset.field_dims[0]
item_num = dataset.field_dims[1]
print("Number of users: ", user_num, ", Number of items: ", item_num)

train_data_n = train_data
val_data_n = test_all_sampling(val_data,  item_num, user_num)
test_data_n = test_all_sampling(test_data,  item_num, user_num)

# Main
args = Argument()
ret = experiment2(dataset,
         train_data_n,
         val_data_n,
         test_data_n,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir,
         args.embed_dim)
