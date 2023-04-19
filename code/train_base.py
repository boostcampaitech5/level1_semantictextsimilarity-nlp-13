import argparse

import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl

from datetime import datetime
import os
import shutil

import yaml
import wandb
from pytorch_lightning.loggers import WandbLogger

from torch.optim.lr_scheduler import LambdaLR

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint


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

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name) #, max_length=160)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
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

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler  = LambdaLR(optimizer, lr_lambda = lambda epoch: 0.95 ** epoch)
        # sch_config = {
        #     "scheduler": scheduler,
        #     "interval": "step",
        # }
        # return {
        #             "optimizer": optimizer,
        #             "lr_scheduler": sch_config,
        #         }
        return{
                    "optimizer": optimizer
                    # "lr_scheduler": scheduler
                }

if __name__ == '__main__':
    wandb.login()
    # 실행 이름 설정
    
    
     
    with open('./config.yaml') as f:
        config = yaml.safe_load(f)


    
    
    model_name= config['model_name']
    project_name = config['project_name']
    wandb_logger = WandbLogger(project=project_name,name=config['display_name'])

    now = datetime.now()
    # model_d = config['dir_name']
    # folder_name = './log/'+ model_d + now.strftime('%Y_%m_%d_%H\'%M\'%S\'/')
    folder_name = './log/'+ project_name+ config['display_name'] + now.strftime('%Y_%m_%d_%H:%M:%S/')
    os.mkdir(folder_name)
    shutil.copyfile('./config.yaml',folder_name+'config.yaml')
    log_dir = folder_name
    os.mkdir(log_dir+'checkpoints/')
    os.mkdir(log_dir+'checkpoints/val/')
    os.mkdir(log_dir+'checkpoints/pearson/')
    os.mkdir(log_dir+'checkpoints/recent/')


    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(config['model_name'], config['train_setting']['batch_size'], config['train_setting']['shuffle'], config['dir']['train_path'], config['dir']['dev_path'],
                            config['dir']['test_path'], config['dir']['predict_path'])

    

    model = Model(config['model_name'], float(config['train_setting']['optimizer']['init_lr']))
    print(model)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min")
    checkpoint_callback_val = ModelCheckpoint(dirpath=log_dir+'checkpoints/val/', save_top_k=2, monitor="val_loss")
    checkpoint_callback_pearson = ModelCheckpoint(dirpath=log_dir+'checkpoints/pearson/', save_top_k=2, monitor="val_pearson",mode='max')
    checkpoint_callback_recent = ModelCheckpoint(dirpath=log_dir+'checkpoints/recent/', save_last=True)


    trainer = pl.Trainer(accelerator='gpu', max_epochs=config['train_setting']['max_epoch'], logger=wandb_logger, log_every_n_steps=1, default_root_dir=folder_name,
                     callbacks = [checkpoint_callback_val,checkpoint_callback_pearson,checkpoint_callback_recent])

    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)
    torch.save(model, folder_name+ 'model.pt')



    model_p = torch.load(log_dir+'model.pt')
    predictions = trainer.predict(model=model_p, datamodule=dataloader)

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = pd.read_csv('./data/sample_submission.csv')
    output['target'] = predictions
    output.to_csv(log_dir+'output.csv', index=False)



    wandb.finish()



    


