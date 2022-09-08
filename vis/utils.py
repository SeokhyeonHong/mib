import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str) # experiment name 
    parser.add_argument('--epoch', type=int, default=0, help='input batch size for training') #epoch num
    parser.add_argument('--index', type=int, help='index of the visualization')
    args = parser.parse_args()
    return args