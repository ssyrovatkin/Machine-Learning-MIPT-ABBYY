import os
import torch


def pytest_generate_tests(metafunc):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    torch.random.manual_seed(42)
