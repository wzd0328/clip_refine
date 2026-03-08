import os
from typing import Any, Tuple

import torchvision.datasets


class GenericDataset(torchvision.datasets.ImageFolder):

    def __init__(self, root, transform, test=False, **kwargs) -> None:
        assert (root is not None) and (transform is not None)
        train = not test
        class_name = self.__class__.__name__
        if train:
            split_data_dir = os.path.join(root, "train")
        else:
            if class_name == 'EuroSAT':
                sub_folder = "val"
            else:
                sub_folder = "test"
            split_data_dir = os.path.join(root, sub_folder)
        super(GenericDataset, self).__init__(root=split_data_dir, transform=transform)


class Aircraft(GenericDataset):

    def __init__(self, transform, test=True, **kwargs):
        super(Aircraft, self).__init__(
            root="/lpai/volumes/so-data-lhp-bd-ga/dataset/zero-shot-datasets/fgvc",
            transform=transform,
            test=test,
        )


class Bird(GenericDataset):

    def __init__(self, transform, test=True, **kwargs):
        super(Bird, self).__init__(
            root="/lpai/volumes/so-data-lhp-bd-ga/dataset/birdsnap/data",
            transform=transform,
            test=test,
        )


class Car(GenericDataset):

    def __init__(self, transform, test=True, **kwargs):
        super(Car, self).__init__(
            root="/lpai/volumes/so-volume-bd-ga/lhp/datasets/Stanford_Cars_dataset",
            transform=transform,
            test=test,
        )


class Caltech101(GenericDataset):

    def __init__(self, transform, test=True, **kwargs):
        super(Caltech101, self).__init__(
            root="/lpai/volumes/so-volume-bd-ga/lhp/datasets/caltech-101",
            transform=transform,
            test=test,
        )


class DTD(GenericDataset):

    def __init__(self, transform, test=True, **kwargs):
        super(DTD, self).__init__(
            root="/lpai/volumes/so-data-lhp-bd-ga/dataset/dtd/data",
            transform=transform,
            test=test,
        )


class EuroSAT(GenericDataset):

    def __init__(self, transform, test=True, **kwargs):
        super(EuroSAT, self).__init__(
            root="/lpai/volumes/so-data-lhp-bd-ga/dataset/zero-shot-datasets/euro_sat",
            transform=transform,
            test=test,
        )


class Food(GenericDataset):

    def __init__(self, transform, test=True, **kwargs):
        super(Food, self).__init__(
            root="/dataset/Food101",
            transform=transform,
            test=test,
        )


class Flower(GenericDataset):

    def __init__(self, transform, test=True, **kwargs):
        super(Flower, self).__init__(
            root="/dataset/OxfordFlower102",
            transform=transform,
            test=test,
        )


class ImageNet(torchvision.datasets.ImageFolder):

    def __init__(self, transform, test=False, **kwargs):
        root = "/lpai/dataset/imagenet-1k/0-1-0"
        train = not test
        if train:
            split_data_dir = os.path.join(root, "train")
        else:
            split_data_dir = os.path.join(root, "ILSVRC2012/val")
        super(ImageNet, self).__init__(root=split_data_dir, transform=transform)


class Pet(GenericDataset):

    def __init__(self, transform, test=True, **kwargs):
        super(Pet, self).__init__(
            root="/dataset/OxfordPets",
            transform=transform,
            test=test,
        )


class SUN397(GenericDataset):

    def __init__(self, transform, test=True, **kwargs):
        super(SUN397, self).__init__(
            root="/dataset/SUN397",
            transform=transform,
            test=test,
        )


class UCF101(GenericDataset):

    def __init__(self, transform, test=True, **kwargs):
        super(UCF101, self).__init__(
            root="/dataset/UCF101",
            transform=transform,
            test=test,
        )
