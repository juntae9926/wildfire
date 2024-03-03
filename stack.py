import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from utils import make_dir, seed_everything
from argparse import ArgumentParser
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

def add_train_args(parser: ArgumentParser):
    parser.add_argument('--n_estimators', default=100, type=int, help='number of trees in the forest')
    parser.add_argument('--save_dir', default='./logs', type=str, help='save dir')
    parser.add_argument('--seed', default=42, type=int, help='seed')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--patience', default=10, type=int, help='patience')
    parser.add_argument('--max_depth', default=6, type=int, help='maximum depth of the tree')
    

def prepare_data(data_blocks, i, val_block, base_dir):
    save_dir = os.path.join(base_dir, str(i))
    
    _data_blocks = data_blocks.copy()
    test_block = _data_blocks.pop(i)
    
    train_blocks = pd.concat(_data_blocks)
    print(f"Lengths of train: {len(train_blocks)}  |  val: {len(val_block)}  |  test: {len(test_block)} | total: {len(train_blocks)+len(val_block)+len(test_block)}")
    val_block["Set"] = 'valid'
    test_block["Set"] = 'test'
    train_blocks["Set"] = 'train'
    data = pd.concat([val_block, test_block, train_blocks]).reset_index()
    return data, save_dir

def process_data(data):
    train_indices = data[data.Set == "train"].index
    valid_indices = data[data.Set == "valid"].index
    test_indices = data[data.Set == "test"].index

    target = 'label'
    features = [col for col in data.columns if col not in ['Set', 'index', target]]

    return data, target, features, train_indices, valid_indices, test_indices

def process_column(data, column):
    data[column] = (data[column] / 1000).apply(lambda x: round(x, 3))
    return data

def train_and_test(data, target, features, train_indices, valid_indices, test_indices, save_dir, args):
    X_train = data[features].iloc[train_indices]
    y_train = data[target].iloc[train_indices]

    X_valid = data[features].iloc[valid_indices]
    y_valid = data[target].iloc[valid_indices]

    X_test = data[features].iloc[test_indices]
    y_test = data[target].iloc[test_indices]

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    # Define base learners
    base_learners = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=args.seed)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=args.seed))
    ]

    # Define meta-learner
    meta_learner = LogisticRegression(random_state=args.seed)

    # Define stacking classifier
    stacking_model = StackingClassifier(estimators=base_learners, final_estimator=meta_learner, cv=5)

    # Fit the model
    stacking_model.fit(X_train_scaled, y_train)

    # Make predictions and evaluate the model
    y_pred = stacking_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy for block {save_dir[-1]}: {test_accuracy}")

    return test_accuracy

def normalize_importance(importance):
    total_importance = sum(importance)
    normalized_importance = [imp / total_importance for imp in importance]
    return normalized_importance

def main(args):
    data_general = pd.read_csv('data/data_general.csv', header=None, skiprows=1, index_col=0)
    data_mountain = pd.read_csv('data/data_mountain.csv', header=None, skiprows=1, index_col=0)
    data = pd.concat([data_general, data_mountain])
    data = data.reset_index(drop=True)

    # Set column names
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

    # Drop unused columns
    data = data.drop(['진입 탈출로 개수', '담 유무', '산림방향 담 유무', '최대 경사도', '최소 경사도'], axis=1)

    # Split data into blocks
    indices = data.index[data['label'].isna()].tolist()
    data_blocks = [data.iloc[start:idx].dropna() for start, idx in zip([0] + indices, indices + [None]) if start != idx]

    # Process specific columns
    for i in [9, 10]:
        data_blocks[i] = process_column(data_blocks[i], '산불 진화용수 거리')
        data_blocks[i] = process_column(data_blocks[i], '소방서와의 거리')

    # Sample validation set from each block
    vals = [block.sample(frac=0.10) for block in data_blocks]
    data_blocks = [block.drop(val.index) for block, val in zip(data_blocks, vals)]
    val_block = pd.concat(vals)

    # Concatenate all data blocks
    data = pd.concat(data_blocks + [val_block])

    # Create save directory
    save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)

    base_dir = make_dir(args.save_dir)
    test_accs = []
    for i in range(len(data_blocks)):
        data, save_dir = prepare_data(data_blocks, i, val_block, base_dir)
        data, target, features, train_indices, valid_indices, test_indices = process_data(data)
        test_acc = train_and_test(data, target, features, train_indices, valid_indices, test_indices, save_dir, args)
        test_accs.append(test_acc)

    print("Average test accuracy:", np.mean(test_accs))
    print(f"Test accuracies: {test_accs}")

if __name__ == "__main__":
    seed_everything(random_seed=42)
    parser = ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args()
    print(args)
    main(args)
