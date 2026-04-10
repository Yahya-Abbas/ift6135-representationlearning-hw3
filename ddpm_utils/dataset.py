import torchvision

try:
    from .args import args
except ImportError:
    from args import args


class FashionMNISTDataset(torchvision.datasets.FashionMNIST):
    def __init__(self, root="."):
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(args.image_size),
                torchvision.transforms.ToTensor(),
            ]
        )

        super().__init__(root, train=True, download=True, transform=transform)

    def __getitem__(self, item):
        return super().__getitem__(item)[0]



MNISTDataset = FashionMNISTDataset
