import numpy as np
import pandas as pd
import cv2
import torch

def parse_boxes(filepath):
    """
    Parse the bounding box file into a pandas dataframe.
    format is a txt file with each line containing:
    track_id, xmin, ymin, xmax, ymax, frame, lost, occuluded, generated, label separated by space
    """
    df = pd.read_csv(filepath, sep=' ', header=None,
                     names=['track_id', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 'lost', 'occluded', 'generated', 'label'])
    return df

def parse_multiple_boxes(filepaths):
    """
    Parse multiple bounding box files into a pandas dataframe.
    format is a txt file with each line containing:
    track_id, xmin, ymin, xmax, ymax, frame, lost, occuluded, generated, label separated by space
    """
    dfs = []
    for path in filepaths:
        df = parse_boxes(path)
        if not df.empty:
            dfs += [df]
        
    return pd.concat(dfs)

def convert_box_coords(coors):
    """
    Convert the bounding box from xmin, ymin, xmax, ymax to x, y, w, h
    """
    xmin, ymin, xmax, ymax = coors
    coors = np.array((xmin, ymin, xmax-xmin, ymax-ymin))
    return coors

def reverse_forecast(history, history_time_steps, future, future_time_step, deg=2):
    last_time_step = history_time_steps[-1]
    history_time_steps += [future_time_step]
    y = np.append(history, future, axis=-1)
    
    x_fun = np.poly1d(np.polyfit(history_time_steps, y[0, 0], deg))
    y_fun = np.poly1d(np.polyfit(history_time_steps, y[0, 1], deg))
    w_fun = np.poly1d(np.polyfit(history_time_steps, y[0, 2], deg))
    h_fun = np.poly1d(np.polyfit(history_time_steps, y[0, 3], deg))
    
    interpolated = np.zeros((4, future_time_step - last_time_step - 1), dtype=np.float32)
    
    inter_range = range(last_time_step + 1, future_time_step)
    
    interpolated[0, :] = x_fun(inter_range)
    interpolated[1, :] = y_fun(inter_range)
    interpolated[2, :] = w_fun(inter_range)
    interpolated[3, :] = h_fun(inter_range)
    
    
    new_history = torch.tensor(interpolated).unsqueeze(0)
    new_history = torch.cat((new_history, future), dim=-1)
    
    return new_history
    
    