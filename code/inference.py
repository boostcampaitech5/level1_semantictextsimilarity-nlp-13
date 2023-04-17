import argparse
import yaml

import pandas as pd

import torch
from model import Dataset, Dataloader, Model
import pytorch_lightning as pl


if __name__ == '__main__':
    # 구현에 사용할 모델이 저장된 폴더 이름을 입력받습니다.
    # 터미널 실행 예시 : python3 run.py --folder_name=2023-04-16_063840
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_name', default="2023-04-16_063840", type=str)
    args = parser.parse_args(args=[])

    folder_name = args.folder_name
    # 하이퍼 파라미터 등 각종 설정값을 yaml 파일을 통해 입력받습니다
    with open(f'./history/{folder_name}/model_config.yaml') as f:
        model_config = yaml.safe_load(f)

    model_parameter = model_config['parameter']
    model_path = model_config['path']

    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(model_config['model_name'], model_parameter['batch_size'], model_parameter['shuffle'], model_path['train'], model_path['dev'],
                            model_path['test'], model_path['predict'])
    model = Model(model_config['model_name'], float(
        model_parameter['learning_rate']))

    # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
    trainer = pl.Trainer(
        accelerator='gpu', max_epochs=model_parameter['max_epoch'], log_every_n_steps=1, default_root_dir=f'history/{folder_name}/')

    # Inference part
    # 저장된 모델로 예측을 진행합니다.
    model = torch.load(f'history/{folder_name}/model.pt')
    predictions = trainer.predict(model=model, datamodule=dataloader)

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = pd.read_csv('./data/sample_submission.csv')
    output['target'] = predictions
    output.to_csv(
        f'history/{folder_name}/{folder_name}_output.csv', index=False)
