import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from refine_pkg.lightning.offsetPredictor_lightning import OffsetPredictorLightning
from refine_pkg.lightning.multiScene_dataset import MultiSceneDataModule
from refine_pkg.configs import get_cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    
    config = get_cfg(args.model)    
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility
    data_module = MultiSceneDataModule(config)
    model = OffsetPredictorLightning(config)
    
    # wandb_logger = WandbLogger(
    #     project="refine", 
    #     name=config.TRAINER.EXP_NAME, 
    #     id=config.TRAINER.EXP_NAME, 
    #     log_model="False", 
    #     save_dir="logs/"
    # )
    
    tensorboard_logger = TensorBoardLogger(
        save_dir="logs_tb/",
        name=config.TRAINER.EXP_NAME,
        version=config.TRAINER.EXP_NAME,  # 用于创建子目录，相当于 WandbLogger 中的 id
        log_graph=False  # 如果你不需要在TensorBoard中记录模型图，可以设置为False
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='AUC_sum',  
        filename='{epoch}-{AUC_sum:.2f}',  
        save_top_k=3,  
        mode='max',  
    )
    
    trainer = pl.Trainer(
        max_epochs=config.TRAINER.EPOCHS,
        accelerator="gpu",
        devices=1 if torch.cuda.is_available() else None,  
        # logger=wandb_logger,
        logger=tensorboard_logger,
        callbacks=[
            LearningRateMonitor(logging_interval="step"), 
            TQDMProgressBar(refresh_rate=1), 
            checkpoint_callback
        ],
        limit_train_batches=1.,
        limit_val_batches=1.,
    )

    # 寻找学习率
    # lr_finder = trainer.tuner.lr_find(model, datamodule=data_module, min_lr=1e-4)
    # print(lr_finder.results)
    # print(f"Suggested Learning Rate: {lr_finder.suggestion()}")
    # lr_finder.plot(suggest=True)
    # plt.savefig("./lr.png")
    
    # 开始训练
    # trainer.fit(model, data_module)
    # load_path = "./logs/refine/akaze_bf/checkpoints/epoch=20-AUC_sum=65.89.ckpt"
    # load_path = "./logs/refine/aliked_lg/checkpoints/epoch=18-AUC_sum=186.74.ckpt"
    # load_path = "./logs/refine/brisk_bf/checkpoints/epoch=21-AUC_sum=78.67.ckpt"
    # load_path = "./logs/refine/disk_lg/checkpoints/epoch=16-AUC_sum=174.96.ckpt"
    # load_path = "./logs/refine/orb_bf/checkpoints/epoch=26-AUC_sum=60.23.ckpt"
    # load_path = "./logs/refine/sift_bf/checkpoints/epoch=18-AUC_sum=88.71.ckpt"
    # load_path = "./logs/refine/sp_lg_1/checkpoints/epoch=18-AUC_sum=187.53.ckpt"
    load_path = "./logs/refine/sp_sg_1/checkpoints/epoch=8-AUC_sum=189.29.ckpt"
    trainer.validate(
        model=model, 
        ckpt_path=load_path,
        datamodule=data_module
    )