import torch
import torch.multiprocessing as mp
import argparse
from model import MyModel
from train import train_main
import numpy as np
import random
import pyfiglet
import sys
import preprocess_minkowski
import test
from colorama import Fore
from config import Config
from utils import load_config_file, get_logger

logger = get_logger("logs/log-main.txt", __name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_data = load_config_file("config.yaml")
cfg = Config(config_data)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def show_menu():
    print(pyfiglet.figlet_format("RPTU     AI Toolkit", font="slant"))
    print(Fore.GREEN + """
================================
  Menu
================================
Commands:
  1. Preprocess Raw Data
  2. Train Model
  3. Test
  4. Inference
  0. Quit
""" + Fore.RESET)

# TODO: add cuda functionality using parser (device = "cuda")
def get_args():
    parser = argparse.ArgumentParser(description='RPTU AI Framework')
    parser.add_argument('-p', '--num-processes', type=int, default=cfg.num_workers,
                        help='how many training processes to use (default: 24)')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='random seed (default: 1)')
    # parser.add_argument('-c','--cuda', action='store_true', help='Enable Cuda')
    return parser.parse_args()

def execute_command(command):
    if command == "1":
        print(Fore.CYAN + "Preprocessing Raw Data..." + Fore.RESET)
        preprocess_minkowski.main()
    elif command == "2":
        print(Fore.CYAN + "Training Model..." + Fore.RESET)
        args = get_args()
        set_seed(args.seed)
        mp.set_start_method('spawn', force=True)
        model = MyModel().to(device)
        train_main()
    elif command == "3":
        print(Fore.CYAN + "Executing Test..." + Fore.RESET)
        try: 
            test.main()
        except:
            print( Fore.RED + "Model not found. Please train the model first, or specify the correct path to the model." + Fore.RESET)
            return
    elif command == "4":
        print(Fore.CYAN + "Executing Inference..." + Fore.RESET)
        print(Fore.CYAN + "Loading Model" + Fore.RESET)
        try:
            model = torch.load(cfg.model_save_path)
            print("Model Loaded")
        except:
            print( Fore.RED + "Model not found. Please train the model first, or specify the correct path to the model." + Fore.RESET)
            return
    else:
        print(Fore.RED + "Invalid command. Just the number suffices. Please try again." + Fore.RESET)
        

def main():
    while True:
        show_menu()
        user_input = input("Enter the command number: ").strip()

        if user_input == "0":
            print(Fore.RED + "Exiting the program" + Fore.RESET)

            sys.exit(0)

        execute_command(user_input)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\n' + Fore.RED + "Manual Interruption!")
        print(Fore.RED + "Quitting...")
        exit(1)
