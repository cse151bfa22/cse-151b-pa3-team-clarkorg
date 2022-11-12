################################################################################
# CSE 151B: Programming Assignment 3
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin, Yash, James
# Fall 2022
################################################################################

from experiment import Experiment
import sys
import torch

def check_cuda():
    '''
    Check if cuda is available
    '''
    print('CUDA is ready for PyTorch:', torch.cuda.is_available())
    print('Number of devices:', torch.cuda.device_count())
    print('Index of device:', torch.cuda.current_device())
    print('Device Name:', torch.cuda.get_device_name(0))


''' Main Driver for your code. Either run `python main.py` which will run the experiment with default config
 or specify the configuration by running `python main.py task-1-default-config` '''
if __name__ == "__main__":

    ''' Check cuda availability'''
    check_cuda()

    ''' Experiment '''
    exp_name = 'task-1.3.5-config'

    # if len(sys.argv) > 1:
    #     exp_name = sys.argv[1]

    print("Running Experiment: ", exp_name)
    exp = Experiment(exp_name)
    exp.test()
    