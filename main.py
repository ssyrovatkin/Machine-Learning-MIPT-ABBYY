import argparse


argument_parser = argparse.ArgumentParser()

argument_parser.add_argument('--mode', type=str, default='train')

if __name__ == '__main__':
    args = argument_parser.parse_args()


