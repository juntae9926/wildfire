import os
from datetime import datetime
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, QuantileTransformer, MinMaxScaler

from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetClassifier

def add_train_args(parser: ArgumentParser):
    parser.add_argument('--seed', default=42, type=int, help='seed')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--save_dir', default='./logs', type=str, help='save dir')
    parser.add_argument('--max_epochs', default=50, type=int, help='max epochs')
    parser.add_argument('--patience', default=10, type=int, help='patience')
    parser.add_argument('--virtual_batch_size', default=256, type=int, help='virtual batch size')
    parser.add_argument('--num_workers', default=12, type=int, help='num workers')
    parser.add_argument('--cat_emb_dim', default=256, type=int, help='cat emb dim')
    parser.add_argument('--mask_type', default='sparsemax', type=str, help='mask type', choices=['sparsemax', 'entmax'])


def seed_everything(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    print(f"Seed locked: {random_seed}")

def process_column(data, column):
    data[column] = (data[column] / 1000).apply(lambda x: round(x, 3))
    return data

def make_dir(base_dir):
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def prepare_data(data_blocks, i, val_block, base_dir):
    save_dir = os.path.join(base_dir, str(i))
    
    _data_blocks = data_blocks.copy()
    test_block = _data_blocks.pop(i)
    
    train_blocks = pd.concat(_data_blocks)
    print(f"lenths of train: {len(train_blocks)}  |  val: {len(val_block)}  |  test: {len(test_block)} | total: {len(train_blocks)+len(val_block)+len(test_block)}")
    val_block["Set"] = 'valid'
    test_block["Set"] = 'test'
    train_blocks["Set"] = 'train'
    data = pd.concat([val_block, test_block, train_blocks]).reset_index()
    return data, save_dir

def process_data(data):
    train_indices = data[data.Set == "train"].index
    valid_indices = data[data.Set == "valid"].index
    test_indices = data[data.Set == "test"].index

    nunique = data.nunique()
    types = data.dtypes

    categorical_columns = []
    categorical_dims = {}

    for col in data.columns:
        if types[col] == 'object' or nunique[col] < 20:
            l_enc = LabelEncoder()
            data[col] = data[col].fillna(0.0)
            data[col] = l_enc.fit_transform(data[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)
        else:
            data.fillna(data.loc[train_indices, col].mean(), inplace=True)

    data.drop(['index'], axis=1, inplace=True)
    target = 'label'
    unused_feat = ['Set']
    features = [col for col in data.columns if col not in unused_feat + [target]]
    cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]
    cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

    return data, target, features, cat_idxs, cat_dims, train_indices, valid_indices, test_indices

from torch.optim.lr_scheduler import ReduceLROnPlateau
def train_and_test(data, target, features, cat_idxs, cat_dims, train_indices, valid_indices, test_indices, save_dir):
    X_train = data[features].values[train_indices]
    y_train = data[target].values[train_indices]

    X_valid = data[features].values[valid_indices]
    y_valid = data[target].values[valid_indices]

    X_test = data[features].values[test_indices]
    y_test = data[target].values[test_indices]

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    ## Train
    clf = TabNetClassifier(cat_idxs=cat_idxs,
                        cat_dims=cat_dims,
                        cat_emb_dim=args.cat_emb_dim,
                        optimizer_fn=torch.optim.Adam,
                        # optimizer_params=dict(lr=args.lr),
                        optimizer_params=dict(lr=args.lr, weight_decay=1e-5),
                        # scheduler_params={"step_size":1, "gamma":0.99},
                        # scheduler_fn=torch.optim.lr_scheduler.StepLR,
                        scheduler_params = dict(mode = "min", patience = 5, min_lr = 1e-5, factor = 0.9),
                        scheduler_fn = ReduceLROnPlateau,
                        mask_type=args.mask_type
                        )

    save_history = []
    clf.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_name=['train', 'valid'],
            eval_metric=['accuracy'],
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            virtual_batch_size=args.virtual_batch_size,
            num_workers=args.num_workers,
            weights=1,
            drop_last=False
            )
    save_history.append(clf.history["valid_accuracy"])
    clf.save_model(os.path.join(save_dir, './ckpt'))

    ## Test
    preds = clf.predict_proba(X_test)
    test_accuracy = accuracy_score(y_true=y_test, y_pred= np.argmax(preds, axis=1))
    print(f"TEST accuracy is {test_accuracy}")

    explain_matrix, masks = clf.explain(X_test, normalize=True)
    weight = np.mean(explain_matrix, axis=0)
    record(save_dir, clf, masks[0], explain_matrix)

    return round(test_accuracy*100, 1), weight

def plot_and_save(data, title, xlabel, ylabel, save_dir, filename, legend=None):
    plt.figure(figsize=(10,6))
    if legend:
        for d, l in zip(data, legend):
            plt.plot(d, label=l)
        plt.legend()
    else:
        plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    save_path = os.path.join(save_dir, filename)  # 저장 경로 수정
    plt.savefig(save_path)

def record(save_dir, clf, masks, explain_matrix):
    # Save masks
    plt.imsave(os.path.join(save_dir, f"mask.png"), masks[10:])  # 저장 경로 수정

    # Save explain matrix
    plt.figure(figsize=(10,6))
    plt.imshow(explain_matrix)
    plt.title("explain_matrix")
    save_path = os.path.join(save_dir, "explain_matrix.png")  # 저장 경로 수정
    plt.savefig(save_path)

    # Save accuracy
    plot_and_save([clf.history['train_accuracy'], clf.history['valid_accuracy']], 'Accuracy over Epochs', 'Epochs', 'Accuracy', save_dir, 'accuracy_plot.png', ['Train Accuracy', 'Validation Accuracy'])

    # Save learning rates
    plot_and_save(clf.history['lr'], 'Learning Rate over Epochs', 'Epochs', 'Learning Rate', save_dir, 'learning_rate_plot.png')

def main(args):
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
        'label'
    ]
    data = data.drop(['진입 탈출로 개수', '담 유무', '산림방향 담 유무', '최대 경사도', '최소 경사도'], axis=1)
    indices = data.index[data['label'].isna()].tolist()

    data_blocks = [data.iloc[start:idx].dropna() for start, idx in zip([0] + indices, indices + [None]) if start != idx]

    for i in [9, 10]:
        data_blocks[i] = process_column(data_blocks[i], '산불 진화용수 거리')
        data_blocks[i] = process_column(data_blocks[i], '소방서와의 거리')

    data_len = [len(block) for block in data_blocks]
    print(data_len)

    vals = [block.sample(frac=0.10) for block in data_blocks]
    data_blocks = [block.drop(val.index) for block, val in zip(data_blocks, vals)]
    val_block = pd.concat(vals)
    data_len = [len(block) for block in data_blocks]
    print(data_len)

    base_dir = make_dir(args.save_dir)
    test_accs = []
    weights = []
    for i in range(len(data_blocks)):
        data, save_dir = prepare_data(data_blocks, i, val_block, base_dir)
        data, target, features, cat_idxs, cat_dims, train_indices, valid_indices, test_indices = process_data(data)
        test_acc, weight = train_and_test(data, target, features, cat_idxs, cat_dims, train_indices, valid_indices, test_indices, save_dir)
        test_accs.append(test_acc)
        weights.append(weight)

    print(f"Test Accuracies: {test_accs}")
    print(f"Average Weights: {np.mean(weights, axis=0):.2f}")
    print(f"Mean Test Accuracy: {sum(test_accs)/len(test_accs):.2f}")


if __name__ == "__main__":
    seed_everything(random_seed=42)
    parser = ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args()
    print(args)
    main(args)
