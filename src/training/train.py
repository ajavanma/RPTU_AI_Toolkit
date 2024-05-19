import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import MinkowskiEngine as ME
from dataset import PointCloudDataset
from config import Config
from pytorch_lightning.callbacks import EarlyStopping
from model import MyModel
from colorama import Fore, Style
from utils import load_config_file, get_logger

logger = get_logger("logs/log-train.txt", __name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_data = load_config_file("config.yaml")
cfg = Config(config_data)

# model is instantiated here, not shared in multiple processes
def train_main() -> None:
    
    train_dataset = PointCloudDataset("train")
    val_dataset = PointCloudDataset("val")
    test_dataset = PointCloudDataset("test")
    
    logger.info(f"trying to load {len(train_dataset)} files from train_dataset")
    logger.info(f"trying to load {len(val_dataset)} files from val_dataloader")
    logger.info(f"trying to load {len(test_dataset)} files from test_dataloader")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        collate_fn=ME.utils.batch_sparse_collate,
        num_workers=cfg.num_workers,
        shuffle=True,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        collate_fn=ME.utils.batch_sparse_collate,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=ME.utils.batch_sparse_collate,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=True
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=cfg.model_save_path,
        filename='rptu-best-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )

    # early stopping when the validation loss does not improve for a certain number of epochs
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    logger.info(f"{Fore.CYAN}Preparing to train...{Style.RESET_ALL}")
    trainer = pl.Trainer(gpus=cfg.num_gpus, callbacks=[checkpoint_callback, early_stopping_callback])
    logger.info(f"{Fore.CYAN}Instantiating Model...{Style.RESET_ALL}")
    model = MyModel()
    logger.info(f"{Fore.CYAN}training... {Style.RESET_ALL}")
    trainer.fit(model, train_dataloader, val_dataloader)
    # how is it different from test.py
    logger.info(f"{Fore.CYAN}Testing... {Style.RESET_ALL}")
    trainer.test(model, test_dataloaders=test_dataloader)
    #torch.save(model.state_dict(), 'model.pth')
    logger.info('saving model to %s', cfg.model_save_path)
    trainer.save_checkpoint(cfg.model_save_path)
    