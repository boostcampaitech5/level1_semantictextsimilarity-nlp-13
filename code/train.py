import yaml
import os

from datetime import datetime

import torch
from model import Dataset, Dataloader, KfoldDataloader, Model
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import wandb


def sweep_train(config=None):
    global sweep_cnt

    wandb.init(config=config, dir=f'./history/{folder_name}/sweep-{sweep_cnt}')
    config = wandb.config

    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(model_config['model_name'], model_parameter['batch_size'], model_parameter['shuffle'], model_path['train'], model_path['dev'],
                            model_path['test'], model_path['predict'])
    model = Model(model_config['model_name'], config.lr)

    # wandb logger 설정
    wandb_logger = WandbLogger(
        project=folder_name, save_dir=f'./history/{folder_name}/')

    # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
    trainer = pl.Trainer(
        accelerator='gpu', max_epochs=model_parameter['max_epoch'], logger=wandb_logger, log_every_n_steps=1, default_root_dir=f'./history/{folder_name}/sweep-{sweep_cnt}/')

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    # sweep cnt 증가
    sweep_cnt += 1


if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 yaml 파일을 통해 입력받습니다.
    with open('./.config/model_config.yaml') as f:
        model_config = yaml.safe_load(f)

    model_parameter = model_config['parameter']
    model_path = model_config['path']
    model_sweep = model_config['sweep']
    model_kfold = model_parameter['kfold']

    # 실험 환경을 저장할 폴더 생성
    now = datetime.now()
    folder_name = now.strftime('%Y-%m-%d_%H%M%S')
    os.mkdir(f'./history/{folder_name}')

    # sweep 여부에 따른 분기
    if model_sweep['isSweep']:
        # sweep별 설정을 저장할 폴더 생성
        for cnt in range(model_sweep['sweepCnt']):
            os.mkdir(f'./history/{folder_name}/sweep-{cnt}')

        # sweep시 필요한 설정값을 yaml 파일을 통해 입력받습니다.
        with open('./.config/sweep_config.yaml') as f:
            sweep_config = yaml.safe_load(f)

        for key, value in sweep_config['parameters'].items():
            if key in ['lr']:
                for k, v in value.items():
                    if k in ['min', 'max', 'mu', 'sigma']:
                        value[k] = float(v)

        sweep_cnt = 0
        sweep_id = wandb.sweep(
            sweep=sweep_config,     # config 딕셔너리를 추가합니다.
            project=folder_name     # project의 이름을 추가합니다.
        )
        wandb.agent(
            sweep_id=sweep_id,              # sweep의 정보를 입력하고
            function=sweep_train,           # train이라는 모델을 학습하는 코드를
            count=model_sweep['sweepCnt']   # 총 n회 실행해봅니다.
        )

        # sweep시 활용한 설정값 저장
        with open(f'./history/{folder_name}/sweep_config.yaml', 'w') as f:
            yaml.dump(sweep_config, f)

    # sweep 없이 훈련 진행
    else:
        # model 생성
        model = Model(model_config['model_name'], float(
            model_parameter['learning_rate']))

        # wandb logger 설정
        wandb_logger = WandbLogger(
            project=folder_name, save_dir=f'./history/{folder_name}/')

        if model_kfold['isKfold']:
            # 평균치를 기록할 리스트
            results = list()

            # num_splits 수 만큼 kfold 수행
            for k in range(model_kfold['num_splits']):
                # kfolddataloader를 생성합니다.
                dataloader = KfoldDataloader(model_config['model_name'], model_parameter['batch_size'], model_parameter['shuffle'], model_path['train'], model_path['dev'],
                                             model_path['test'], model_path['predict'], k, model_kfold['split_seed'], model_kfold['num_splits'])
                dataloader.prepare_data()
                dataloader.setup()

                # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
                trainer = pl.Trainer(
                    accelerator='gpu', max_epochs=model_parameter['max_epoch'], logger=wandb_logger, log_every_n_steps=1, default_root_dir=f'./history/{folder_name}/')

                # Train part
                trainer.fit(model=model, datamodule=dataloader)
                score = trainer.test(model=model, datamodule=dataloader)

                results.extend(score)

            # kfold 수행 후 평균 성능 기록
            result = [x['test_pearson'] for x in results]
            score = sum(result) / model_kfold['num_splits']
            result.append(score)
            wandb_logger.log_text(key='kfold result', columns=[i for i in range(
                model_kfold['num_splits'])]+['avg'], data=[result])

        else:
            # dataloader와 model을 생성합니다.
            dataloader = Dataloader(model_config['model_name'], model_parameter['batch_size'], model_parameter['shuffle'], model_path['train'], model_path['dev'],
                                    model_path['test'], model_path['predict'])
            model = Model(model_config['model_name'], float(
                model_parameter['learning_rate']))

            # wandb logger 설정
            if model_config['project_name']:
                wandb_logger = WandbLogger(
                    project=model_config['project_name'], name=model_config['display_name'], save_dir=f'./history/{folder_name}/')
            else:
                wandb_logger = WandbLogger(
                    project=folder_name, save_dir=f'./history/{folder_name}/')

            lr_monitor = LearningRateMonitor(logging_interval='step')
            checkpoint_callback_loss = ModelCheckpoint(
                dirpath=f'./history/{folder_name}/checkpoints/loss/', save_top_k=2, monitor="val_loss")
            checkpoint_callback_pearson = ModelCheckpoint(
                dirpath=f'./history/{folder_name}/checkpoints/pearson/', save_top_k=2, monitor="val_pearson", mode='max')

            # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
            trainer = pl.Trainer(
                accelerator='gpu', max_epochs=model_parameter['max_epoch'], logger=wandb_logger, log_every_n_steps=1, default_root_dir=f'./history/{folder_name}/', callbacks=[lr_monitor, checkpoint_callback_loss, checkpoint_callback_pearson])

            # Train part
            trainer.fit(model=model, datamodule=dataloader)
            trainer.test(model=model, datamodule=dataloader)

        # 학습이 완료된 모델을 저장합니다.
        torch.save(model, f'./history/{folder_name}/model.pt')

    # 학습에 활용한 설정값 저장
    with open(f'./history/{folder_name}/model_config.yaml', 'w') as f:
        yaml.dump(model_config, f)
