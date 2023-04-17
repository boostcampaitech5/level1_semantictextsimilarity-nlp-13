import yaml
import os

import pandas as pd

from tqdm.auto import tqdm
from datetime import datetime

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import wandb


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)


class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, max_length=160)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = '[SEP]'.join([item[text_column]
                                for text_column in self.text_columns])
            outputs = self.tokenizer(
                text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
        return data

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            # 학습데이터 준비
            train_inputs, train_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)


class Model(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1)
        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        self.loss_func = torch.nn.L1Loss()

    def forward(self, x):
        x = self.plm(x)['logits']

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(
            logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(
            logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, eps=1e-08)
        return optimizer


def sweep_train(config=None):
    global sweep_cnt

    wandb.init(config=config, dir=f'history/{folder_name}/sweep-{sweep_cnt}/')
    config = wandb.config

    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(model_config['model_name'], model_parameter['batch_size'], model_parameter['shuffle'], model_path['train'], model_path['dev'],
                            model_path['test'], model_path['predict'])
    model = Model(model_config['model_name'], config.lr)

    # wandb logger 설정
    wandb_logger = WandbLogger(project=folder_name)

    # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
    trainer = pl.Trainer(
        accelerator='gpu', max_epochs=model_parameter['max_epoch'], logger=wandb_logger, log_every_n_steps=1, default_root_dir=f'history/{folder_name}/sweep-{sweep_cnt}/')

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
            if 'min' in value:
                value['min'] = float(value['min'])
            if 'max' in value:
                value['max'] = float(value['max'])
            if 'mu' in value:
                value['mu'] = float(value['mu'])
            if 'sigma' in value:
                value['sigma'] = float(value['sigma'])

        sweep_cnt = 0
        sweep_id = wandb.sweep(
            sweep=sweep_config,                 # config 딕셔너리를 추가합니다.
            project=folder_name  # project의 이름을 추가합니다.
        )
        wandb.agent(
            sweep_id=sweep_id,              # sweep의 정보를 입력하고
            function=sweep_train,           # train이라는 모델을 학습하는 코드를
            count=model_sweep['sweepCnt']   # 총 n회 실행해봅니다.
        )

        # sweep시 활용한 설정값 저장
        with open(f'./history/{folder_name}/sweep_config.yaml', 'w') as f:
            yaml.dump(sweep_config, f)

    else:
        # dataloader와 model을 생성합니다.
        dataloader = Dataloader(model_config['model_name'], model_parameter['batch_size'], model_parameter['shuffle'], model_path['train'], model_path['dev'],
                                model_path['test'], model_path['predict'])
        model = Model(model_config['model_name'], float(
            model_parameter['learning_rate']))

        # wandb logger 설정
        wandb_logger = WandbLogger(project=folder_name)

        # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
        trainer = pl.Trainer(
            accelerator='gpu', max_epochs=model_parameter['max_epoch'], logger=wandb_logger, log_every_n_steps=1, default_root_dir=f'history/{folder_name}/')

        # Train part
        trainer.fit(model=model, datamodule=dataloader)
        trainer.test(model=model, datamodule=dataloader)

        # 학습이 완료된 모델을 저장합니다.
        torch.save(model, f'./history/{folder_name}/model.pt')

    # 학습에 활용한 설정값 저장
    with open(f'./history/{folder_name}/model_config.yaml', 'w') as f:
        yaml.dump(model_config, f)
