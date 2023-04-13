import yaml

if __name__ == '__main__':
    with open('./config.yaml') as f:
        config = yaml.safe_load(f)
    print(config)
    print()
    print(config['model_name'])
    print(config['train_setting'])
    print(config['train_setting']['optimizer']['type'])
    print(config['dir'])
    