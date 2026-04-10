import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def GetTransform():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])
    return train_transform,test_transform

def GetCifar_10(BATCH_SIZE=128,num_workers=8):
    train_transform,test_transform=GetTransform()

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform
    )

    # 先跑通，Windows 下先别开多进程
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers
    )
    return train_loader,test_loader

def GetCifar_100(BATCH_SIZE=128,num_workers=8):
    train_transform,test_transform=GetTransform()

    train_dataset = datasets.CIFAR100(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR100(
        root="./data",
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers
    )
    return train_loader,test_loader


class CIFAR100Coarse(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, download=True):
        super().__init__(root=root, train=train, transform=transform, download=download)
        #关键：重新加载 coarse label
        if train:
            self.targets = self._load_coarse_labels("train")
        else:
            self.targets = self._load_coarse_labels("test")

    def _load_coarse_labels(self, split):
        import pickle, os

        file = os.path.join(self.root, "cifar-100-python", split)
        with open(file, "rb") as f:
            entry = pickle.load(f, encoding="latin1")

        # print('11213123',len(entry["coarse_labels"]))

        return entry["coarse_labels"]

##coarse labels
# coarse_classes = [
#     "aquatic_mammals",
#     "fish",
#     "flowers",
#     "food_containers",
#     "fruit_and_vegetables",
#     "household_electrical_devices",
#     "household_furniture",
#     "insects",
#     "large_carnivores",
#     "large_man-made_outdoor_things",
#     "large_natural_outdoor_scenes",
#     "large_omnivores_and_herbivores",
#     "medium_mammals",
#     "non-insect_invertebrates",
#     "people",
#     "reptiles",
#     "small_mammals",
#     "trees",
#     "vehicles_1",
#     "vehicles_2",
# ]

def GetCifar_100_Coarse(BATCH_SIZE=128,num_workers=8):
    train_transform,test_transform=GetTransform()

    train_dataset = CIFAR100Coarse(root="./data", train=True, transform=train_transform)
    test_dataset  = CIFAR100Coarse(root="./data", train=False, transform=test_transform)

    # print(coarse_labels = train_dataset.targets_coarse)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers
    )
    return train_loader,test_loader

class CIFAR100Super5(CIFAR100Coarse):
    def __init__(self, root, train=True, transform=None, download=True):
        super().__init__(root=root, train=train, transform=transform, download=download)

        # 20个 coarse 类 -> 5个 super 类
        self.coarse_to_super = {
            0: 0,   # aquatic_mammals -> animals
            1: 0,   # fish -> animals
            2: 2,   # flowers -> plants
            3: 3,   # food_containers -> manmade_objects
            4: 2,   # fruit_and_vegetables -> plants
            5: 3,   # household_electrical_devices -> manmade_objects
            6: 3,   # household_furniture -> manmade_objects
            7: 0,   # insects -> animals
            8: 0,   # large_carnivores -> animals
            9: 3,   # large_man-made_outdoor_things -> manmade_objects
            10: 4,  # large_natural_outdoor_scenes -> natural_scenes
            11: 0,  # large_omnivores_and_herbivores -> animals
            12: 0,  # medium_mammals -> animals
            13: 0,  # non-insect_invertebrates -> animals
            14: 1,  # people -> people
            15: 0,  # reptiles -> animals
            16: 0,  # small_mammals -> animals
            17: 2,  # trees -> plants
            18: 3,  # vehicles_1 -> manmade_objects
            19: 3,  # vehicles_2 -> manmade_objects
        }

        self.super_classes = [
            "animals",
            "people",
            "plants",
            "manmade_objects",
            "natural_scenes"
        ]

        # 直接把 targets 替换成 5 类标签
        self.targets = [self.coarse_to_super[t] for t in self.targets]

def GetCifar_100_Super5(BATCH_SIZE=128, num_workers=8):
    train_transform, test_transform = GetTransform()

    train_dataset = CIFAR100Super5(
        root="./data",
        train=True,
        transform=train_transform,
        download=True
    )

    test_dataset = CIFAR100Super5(
        root="./data",
        train=False,
        transform=test_transform,
        download=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader

if __name__=='__main__':
    GetCifar_100_Coarse()