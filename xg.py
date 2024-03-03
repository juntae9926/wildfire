import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from utils import make_dir, seed_everything
from argparse import ArgumentParser
from utils import save_correlation_heatmap
from utils import compute_shap_values

def add_train_args(parser: ArgumentParser):
    parser.add_argument('--n_estimators', default=100, type=int, help='number of trees in the forest')
    parser.add_argument('--max_depth', default=None, type=int, help='maximum depth of the tree')
    parser.add_argument('--save_dir', default='./logs', type=str, help='save dir')
    parser.add_argument('--seed', default=42, type=int, help='seed')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--patience', default=10, type=int, help='patience')
    

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

    xgb_model = XGBClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.lr, random_state=args.seed)
    xgb_model.fit(X_train_scaled, y_train, early_stopping_rounds=args.patience, eval_set=[(X_valid_scaled, y_valid)], verbose=False)

    y_pred = xgb_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Test accuracy for block {save_dir[-1]}: {test_accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    feature_importances = xgb_model.feature_importances_
    print("Feature importances:")
    for feature, importance in zip(features, feature_importances):
        print(f"{feature}: {importance}")


    return test_accuracy, precision, recall, f1, feature_importances

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
    # data = data.drop(['진입 탈출로 개수', '교차로 개수', '산림방향 담 유무', '최대 경사도', '최소 경사도'], axis=1)

    column_name_mapping_corrected = {
        # '진입 탈출로 개수': 'The number of Escape Road',
        '30m 이내 진입 탈출로 개수': 'The number of Escape Road',
        '최소 도로폭': 'Road Width',
        '교차로 개수': 'The number of Intersection',
        '침엽수림 비율': 'Coniferous Forest Ratio',
        '산림기준 이격 거리': 'Forest Distance',
        '비산거리': 'Scattering Distance',
        '창문': 'Building Area',
        '주건물 지붕특성': 'Building Roof Characteristics',
        '산불 진화용수 거리': 'Reservoir Distance',
        '소방서와의 거리': 'Fire Station Distance',
        '담 유무': 'Fence'
    }
    data = data.rename(columns=column_name_mapping_corrected)

    # Split data into blocks
    indices = data.index[data['label'].isna()].tolist()
    data_blocks = [data.iloc[start:idx].dropna() for start, idx in zip([0] + indices, indices + [None]) if start != idx]

    # Process specific columns
    for i in [9, 10]:
        data_blocks[i] = process_column(data_blocks[i], 'Reservoir Distance')
        data_blocks[i] = process_column(data_blocks[i], 'Fire Station Distance')

    # Sample validation set from each block
    vals = [block.sample(frac=0.10) for block in data_blocks]
    data_blocks = [block.drop(val.index) for block, val in zip(data_blocks, vals)]
    val_block = pd.concat(vals)

    # Concatenate all data blocks
    data = pd.concat(data_blocks + [val_block])

    save_correlation_heatmap(data, 'correlation_heatmap.png')

    # Create save directory
    save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)


    base_dir = make_dir(args.save_dir)
    test_accs = []
    precisions = []
    recalls = []
    f1_scores = []
    feature_importances_list = []
    for i in range(len(data_blocks)):
        data, save_dir = prepare_data(data_blocks, i, val_block, base_dir)
        data, target, features, train_indices, valid_indices, test_indices = process_data(data)
        test_acc, precision, recall, f1, feature_importances = train_and_test(data, target, features, train_indices, valid_indices, test_indices, save_dir, args)
        test_accs.append(test_acc)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        feature_importances_list.append(feature_importances)

    print("Average test accuracy:", np.mean(test_accs))
    print(f"Test accuracies: {test_accs}")
    print("Average precision:", np.mean(precisions))
    print("Average recall:", np.mean(recalls))
    print("Average f1 score:", np.mean(f1_scores))
    print("Average feature importances:")
    avg_feature_importances = np.mean(feature_importances_list, axis=0)
    for feature, importance in zip(features, avg_feature_importances):
        print(f"{feature}: {importance}")

if __name__ == "__main__":
    seed_everything(random_seed=42)
    parser = ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args()
    print(args)
    main(args)
