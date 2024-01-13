import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

from matplotlib import pyplot as plt

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

MAX_EPOCHS = 5

def seed_everything(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    print(f"Seed locked: {random_seed}")

seed_everything(random_seed=42)

# Load the dataset from the CSV file
# data = pd.read_csv('data_230817.csv', header=None)
# data = pd.read_csv('data_230916.csv', header=None, skiprows=1, index_col=0)
# data.columns = [
#     '진입 탈출로 개수',
#     '30m 이내 진입 탈출로 개수',
#     '최소 도로폭',
#     '교차로 개수',
#     '침엽수림 비율',
#     '산림기준 이격 거리',
#     '창문',
#     '산불 진화용수 거리',
#     '소방서와의 거리',
#     '주건물 지붕특성',
#     '불난 후 상태'
# ]


data_general = pd.read_csv('data/data_general.csv', header=None, skiprows=1, index_col=0)
data_mountain = pd.read_csv('data/data_mountain.csv', header=None, skiprows=1, index_col=0)
data = pd.concat([data_general, data_mountain])
data = data.reset_index(drop=True)

data.columns = [
    '진입 탈출로 개수',
    '30m 이내 진입 탈출로 개수',
    '최소 도로폭',
    '교차로 개수',
    '침엽수림 비율',
    '산림기준 이격 거리',
    '창문',
    '산불 진화용수 거리',
    '소방서와의 거리',
    '담 유무',
    '산림방향 담 유무',
    '최대 경사도',
    '최소 경사도',
    '주건물 지붕특성',
    '비산거리',
    '불난 후 상태'
]
data = data.drop(['진입 탈출로 개수', '담 유무', '산림방향 담 유무', '최대 경사도', '최소 경사도'], axis=1)
# data = data.drop(['교차로 개수', '산불 진화용수 거리', '소방서와의 거리', '담 유무', '산림방향 담 유무', '최대 경사도', '최소 경사도'], axis=1)

# 마을 별 구분
# indices = data.index[data.isna().any(axis=1)].tolist()
indices = data.index[data['불난 후 상태'].isna()].tolist()
data_blocks = []
start = 0
for idx in indices:
    if idx == 0:
        continue
    data_blocks.append(data.iloc[start:idx])
    start = idx + 1

if start < len(data):
    data_blocks.append(data.iloc[start:])

data_blocks = [block.dropna() for block in data_blocks]

# 9,10 마을 제외
# a = data_blocks.pop(-1)
# b = data_blocks.pop(-1)
# data_blocks = []
# data_blocks.append(a)
# data_blocks.append(b)

data_blocks[9]['산불 진화용수 거리'] = data_blocks[9]['산불 진화용수 거리'] / 1000
data_blocks[9]['산불 진화용수 거리'] = data_blocks[9]['산불 진화용수 거리'].apply(lambda x: round(x, 3))

data_blocks[10]['산불 진화용수 거리'] = data_blocks[10]['산불 진화용수 거리'] / 1000
data_blocks[10]['산불 진화용수 거리'] = data_blocks[10]['산불 진화용수 거리'].apply(lambda x: round(x, 3))

data_blocks[9]['소방서와의 거리'] = data_blocks[9]['소방서와의 거리'] / 1000
data_blocks[9]['소방서와의 거리'] = data_blocks[9]['소방서와의 거리'].apply(lambda x: round(x, 3))

data_blocks[10]['소방서와의 거리'] = data_blocks[10]['소방서와의 거리'] / 1000
data_blocks[10]['소방서와의 거리'] = data_blocks[10]['소방서와의 거리'].apply(lambda x: round(x, 3))

data_len = [len(block) for block in data_blocks]
print(data_len)

# 산림 기준 학습 split
# split = [64.17, 38.0, 35.52, 53.92, 57.39, 67.93, 83.0, 22.67, 23.27, 34.0, 32.63]
# over = [split.index(x) for x in split if x > 50.0]
# under = [split.index(x) for x in split if x < 50.0]

# over
# data_blocks = [data_blocks[i] for i in over]
# data_blocks.pop(4)
# data_blocks.pop(6)
# data_blocks.pop(6)
# data_len = [len(block) for block in data_blocks]
# print(data_len)

# o = data_blocks.pop(0)
# a = data_blocks.pop(3)
# b = data_blocks.pop(5)
# c = data_blocks.pop(5)
# data_blocks = []
# data_blocks.append(o)
# data_blocks.append(a)
# data_blocks.append(b)
# data_blocks.append(c)

# data_len = [len(block) for block in data_blocks]
# print(data_len)


# Random set 구분
# data = pd.concat(data_blocks).reset_index()
# if "Set" not in data.columns:
    # data["Set"] = np.random.choice(["train", "valid", "test"], p =[.8, .1, .1], size=(data.shape[0]))


# validation 데이터를 전체의 10퍼센트씩으로 구성할 경우
vals = []
for idx in range(len(data_blocks)):
    val = data_blocks[idx].sample(frac=0.10)
    data_blocks[idx].drop(val.index, inplace=True)
    vals.append(val)
val_block = pd.concat(vals)
data_len = [len(block) for block in data_blocks]
print(data_len)


# 학습
test_accs = []
weights = []
for i in range(len(data_blocks)):
    # if not i > 8:
    #     continue

    save_dir = f'figs/{i}/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    _data_blocks = data_blocks.copy()
    # val_block = data_blocks.pop(0)
    test_block = _data_blocks.pop(i)
    
    train_blocks = pd.concat(_data_blocks)
    print(f"lenths of train: {len(train_blocks)}  |  val: {len(val_block)}  |  test: {len(test_block)} | total: {len(train_blocks)+len(val_block)+len(test_block)}")
    val_block["Set"] = 'valid'
    test_block["Set"] = 'test'
    train_blocks["Set"] = 'train'
    data = pd.concat([val_block, test_block, train_blocks]).reset_index()
    

    train_indices = data[data.Set=="train"].index
    valid_indices = data[data.Set=="valid"].index
    test_indices = data[data.Set=="test"].index

    nunique = data.nunique()
    types = data.dtypes

    categorical_columns = []
    categorical_dims = {}

    for col in data.columns:
        if types[col] == 'object' or nunique[col] < 20:
            # print(col, data[col].nunique())
            l_enc = LabelEncoder()
            data[col] = data[col].fillna(0.0)
            data[col] = l_enc.fit_transform(data[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)
        
        else:
            data.fillna(data.loc[train_indices, col].mean(), inplace=True)

    data.drop(['index'], axis=1, inplace=True)
    target = '불난 후 상태'
    unused_feat = ['Set']
    features = [col for col in data.columns if col not in unused_feat+[target]]
    cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]
    cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

    X_train = data[features].values[train_indices]
    y_train = data[target].values[train_indices]

    X_valid = data[features].values[valid_indices]
    y_valid = data[target].values[valid_indices]

    X_test = data[features].values[test_indices]
    y_test = data[target].values[test_indices]

    ## Train
    clf = TabNetClassifier(cat_idxs=cat_idxs,
                        cat_dims=cat_dims,
                        cat_emb_dim=64,
                        optimizer_fn=torch.optim.Adam,
                        optimizer_params=dict(lr=2e-2),
                        scheduler_params={"step_size":1,
                                            "gamma":0.99},
                        scheduler_fn=torch.optim.lr_scheduler.StepLR,
                        mask_type='entmax' # "sparsemax", entmax)
                        )


    save_history = []
    clf.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_name=['train', 'valid'],
            eval_metric=['accuracy'],
            max_epochs=MAX_EPOCHS,
            patience=20,
            batch_size=256,
            virtual_batch_size=256,
            num_workers=12,
            weights=1,
            drop_last=False
            )
    save_history.append(clf.history["valid_accuracy"])
    clf.save_model('./ckpt')


    ## Test
    preds = clf.predict_proba(X_test)
    # test_auc = roc_auc_score(y_score=preds[:,1], y_true=y_test)
    test_accuracy = accuracy_score(y_true=y_test, y_pred= np.argmax(preds, axis=1))
    test_accs.append(round(test_accuracy*100, 1))
    print(f"TEST accuracy is {test_accuracy}")

    explain_matrix, masks = clf.explain(X_test, normalize=True)
    weight = np.mean(explain_matrix, axis=0)
    weights.append(weight)
    

    fig, axs = plt.subplots(1, 3, figsize=(20,20))
    for i in range(3):
        axs[i].imshow(masks[i][:50])
        axs[i].set_title(f"mask {i}")
    fig.savefig("masks.png")

    # plot explain matrix
    plt.figure(figsize=(10,6))
    plt.imshow(explain_matrix)
    plt.title("explain_matrix")
    plt.savefig(save_dir + "explain_matrix.png")

    # plot losses
    plt.figure(figsize=(10,6))
    plt.plot(clf.history['loss'], label='Train loss')
    plt.title("Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_dir +  "loss_plot.png")

    # plot auc
    plt.figure(figsize=(10,6))  # Create a new figure
    plt.plot(clf.history['train_accuracy'], label='Train Accuracy')
    plt.plot(clf.history['valid_accuracy'], label='Validation Accuracy')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(save_dir + "accuracy_plot.png")

    # plot learning rates
    plt.figure(figsize=(10,6))  # Create another new figure
    plt.plot(clf.history['lr'])
    plt.title("Learning Rate over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.savefig(save_dir + "learning_rate_plot.png")

print(test_accs)
print(f"weights are {np.mean(weights, axis=0)}")
print(f"MEAN is {sum(test_accs)/len(test_accs)}")