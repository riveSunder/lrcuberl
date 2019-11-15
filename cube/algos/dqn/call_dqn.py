import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str)

args = parser.parse_args()

exp_name = args.exp_name

epochs = 4
max_epochs = 20

for start_epoch in range(0, max_epochs, epochs):
    os.system("python ./cube/algos/dqn/dqn.py --epochs {} --exp_name {} --start {}"\
            .format(epochs, exp_name, start_epoch))


