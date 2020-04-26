import yaml


def get(file):
    with open(file, 'r') as r:
        loader = yaml.load(r, Loader=yaml.FullLoader)
        return loader


if __name__ == '__main__':
    r = get('../configuration/feature.yaml')
    print(r)
    print(r['label'])
    print(list(i for i in r['category']))
