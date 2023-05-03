import numpy as np
import pandas as pd
import cv2
import torch
from sklearn.model_selection import train_test_split

from utils import *

class ForecastDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def create_training_data(boxes_df, history_size, forecast_size, train_split=0.8, dataset_size=None, save=False, save_folder=None):
    """
    Create time series forecast training data
    input_size: (history_size, 4)
    label_size: (history_size, 4)
    """
    coordinates = boxes_df.loc[:, ['xmin', 'ymin', 'xmax', 'ymax']].values
    
    coordinates = np.array([convert_box_coords(coord) for coord in coordinates])
    
    x, y = [], []
    
    for i in range(len(coordinates) - history_size - forecast_size):
        x.append(coordinates[i:i + history_size])
        y.append(coordinates[i + history_size: i + history_size + forecast_size])
    
    x = np.array(x)
    y = np.array(y)
    # permute the dims to (batch_size, 4, history_size)
    x = np.transpose(x, (0, 2, 1))
    y = np.transpose(y, (0, 2, 1))
    
    if dataset_size:
        dataset_size = int(dataset_size * len(x))
        rand_idx = np.random.choice(len(x), replace=False, size=dataset_size)
        
        x, y = x[rand_idx], y[rand_idx]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_split)
    
    # print('x_train shape:', x_train.shape)
    # print('y_train shape:', y_train.shape)
    # print('x_test shape:', x_test.shape)
    # print('y_test shape:', y_test.shape)
    
    if save:
        save_folder = save_folder if save_folder else 'data'
        np.save(f'{save_folder}/x_train.npy', x_train)
        np.save(f'{save_folder}/y_train.npy', y_train)
        np.save(f'{save_folder}/x_test.npy', x_test)
        np.save(f'{save_folder}/y_test.npy', y_test)
    
    return x_train, y_train, x_test, y_test

def create_video_test(boxes_df, history_size, forecast_size, save=False, save_folder=None):
    """
    Create time series forecast training data
    input_size: (history_size, 4)
    label_size: (history_size, 4)
    """
    coordinates = boxes_df.loc[:, ['xmin', 'ymin', 'xmax', 'ymax']].values
    
    coordinates = np.array([convert_box_coords(coord) for coord in coordinates])
    
    x = coordinates[:history_size]
    y = coordinates[history_size:]
    
    x = np.array(x)
    y = np.array(y)
    x = x[np.newaxis, :, :]
    y = y[np.newaxis, :, :]
    # permute the dims to (batch_size, 4, history_size)
    x = np.transpose(x, (0, 2, 1))
    y = np.transpose(y, (0, 2, 1))
    
    print(x.shape)
    print(y.shape)
    
    if save:
        save_folder = save_folder if save_folder else 'data'
        np.save(f'{save_folder}/x_test.npy', x)
        np.save(f'{save_folder}/y_test.npy', y)
    
    return x, y

def create_dataloader(features, labels, batch_size=1):
    train_ds = ForecastDataset(features, labels)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    return train_dl
    
if __name__ == '__main__':
    
    
    # data = ['/Users/abhiram/Documents/NFS/basketball_player_2/240/basketball_player_2.txt',
    #         '/Users/abhiram/Documents/NFS/Gymnastics/240/Gymnastics.txt',
    #         '/Users/abhiram/Documents/NFS/soccer_player_3/240/soccer_player_3.txt',
    #         '/Users/abhiram/Documents/NFS/running_2/240/running_2.txt',
    #         '/Users/abhiram/Documents/NFS/walking_3/240/walking_3.txt',
    #         '/Users/abhiram/Documents/NFS/walking/240/walking.txt',
    #         '/Users/abhiram/Documents/NFS/basketball_player/240/basketball_player.txt',
    #         # 'data/NFS/airplane_landing/240/airplane_landing.txt',
    #         # 'data/NFS/airboard_1/30/airboard_1.txt',
    #         # 'data/NFS/airboard_1/240/airboard_1.txt'
    #         ]
    # x_train, y_train, x_test, y_test = [], [], [], []
    # for i in data:
    #     boxes_df = parse_boxes(i)
    #     boxes2 = boxes_df.reindex(index=boxes_df.index[::-1]) 
    #     x1, y1, x2, y2 = create_training_data(boxes_df, history_size=10, forecast_size=1, train_split=0.8, dataset_size=0.5, save=False, save_folder='data')
    #     x3, y3, x4, y4 = create_training_data(boxes2, history_size=10, forecast_size=1, train_split=0.8, dataset_size=0.5, save=False, save_folder='data')
    #     x_train.extend([x1, x3])
    #     y_train.extend([y1, y3])
    #     x_test.extend([x2, x4])
    #     y_test.extend([y2, y4])
    
    # x_train, y_train, x_test, y_test = np.concatenate(x_train), np.concatenate(y_train), np.concatenate(x_test), np.concatenate(y_test)
    # print('x_train shape:', x_train.shape)
    # print('y_train shape:', y_train.shape)
    # print('x_test shape:', x_test.shape)
    # print('y_test shape:', y_test.shape)
    # save_folder='data/box_training_data'
    # np.save(f'{save_folder}/x_train.npy', x_train)
    # np.save(f'{save_folder}/y_train.npy', y_train)
    # np.save(f'{save_folder}/x_test.npy', x_test)
    # np.save(f'{save_folder}/y_test.npy', y_test)
    # create_training_data(boxes_df, history_size=20, train_split=0.8, dataset_size=None, save=True, save_folder='data/box_training_data')
    
    data = ['/Users/abhiram/Documents/NFS/basketball_player/240/basketball_player.txt']
    boxes_df = parse_boxes(data[0])
    x, y = create_video_test(boxes_df, history_size=10, forecast_size=None, save=True, save_folder='data/test_video')
    
    