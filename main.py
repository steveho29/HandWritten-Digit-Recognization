from init import*


model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)),
    nn.ReLU(),
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Dropout(0.25),

    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
    nn.ReLU(),
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Dropout(0.25),

    nn.Linear(128 * 3 * 3, 32),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(32, 10),
    nn.LogSoftmax(dim=1)
)


def get_data():
    batch_size = 64
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = datasets.MNIST('MNIST/', download=False, train=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    return train_loader


def run():
    data = get_data()
    print(len(data))

run()