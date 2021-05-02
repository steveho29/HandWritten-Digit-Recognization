import torch, torchvision, numpy
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image, ImageFilter


class MnistModel(nn.Module):
    def __init__(self, classes: int) -> None:
        super(MnistModel, self).__init__()

        self.classes = classes

        # initialize the layers in the first (CONV => RELU) * 2 => POOL + DROP
        # (N,1,28,28) -> (N,16,24,24)
        self.conv1A = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        # (N,16,24,24) -> (N,32,20,20)
        self.conv1B = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        # (N,32,20,20) -> (N,32,10,10)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.act = nn.ReLU()
        self.do = nn.Dropout(0.25)

        # initialize the layers in the second (CONV => RELU) * 2 => POOL + DROP
        # (N,32,10,10) -> (N,64,8,8)
        self.conv2A = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        # (N,64,8,8) -> (N,128,6,6)
        self.conv2B = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        # (N,128,6,6) -> (N,128,3,3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # initialize the layers in our fully-connected layer set
        # (N,128,3,3) -> (N,32)
        self.dense3 = nn.Linear(128*3*3, 32)

        # initialize the layers in the softmax classifier layer set
        # (N, classes)
        self.dense4 = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # build the first (CONV => RELU) * 2 => POOL layer set
        x = self.conv1A(x)
        x = self.act(x)
        x = self.conv1B(x)
        x = self.act(x)
        x = self.pool1(x)
        x = self.do(x)

        # build the second (CONV => RELU) * 2 => POOL layer set
        x = self.conv2A(x)
        x = self.act(x)
        x = self.conv2B(x)
        x = self.act(x)
        x = self.pool2(x)
        x = self.do(x)

        # build our FC layer set
        x = x.view(x.size(0), -1)
        x = self.dense3(x)
        x = self.act(x)
        x = self.do(x)

        # build the softmax classifier
        x = nn.functional.log_softmax(self.dense4(x), dim=1)
        return x


def show_image(loader: DataLoader):
    num_of_images = 64
    i = 0
    for images, labels in loader:
        plt.figure(figsize=(8, 20))
        for index in range(0, num_of_images):
            plt.subplot(10, 10, index + 1)
            plt.axis('off')
            plt.title(labels.storage()[index])
            plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
            # torchvision.utils.save_image(images[index], 'IMAGE/' + str(i) + '.png')
            i += 1
        # plt.show()


def explore_data():
    batch_size = 64
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_set = datasets.MNIST('MNIST/', download=False, train=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = datasets.MNIST('MNIST/', download=False, train=False, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    show_image(train_loader)


def training(model_path):
    model = nn.Sequential(
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.Tanh(),
        nn.Linear(64, 10),
        nn.LogSoftmax(dim=1)
    )
    model = torch.load(model_path)
    batch_size = 64
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = datasets.MNIST('MNIST/', download=False, train=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    data_iter = iter(train_loader)  # creating a iterator

    epochs = 100
    running_lost_list = []
    epochs_list = []

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    # criterion = nn.MSELoss()
    for epoch in range(epochs):
        running_lost = 0
        for images, labels in train_loader:
            print(images.shape)
            images = images.view(images.shape[0], -1)
            print(images.shape)
            break
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_lost += loss.item()

        print("Epoch {} - Training Loss: {}".format(epoch, running_lost / len(train_loader)))
    torch.save(model, 'linear_model.pt')


def classify(img, ps):
    ps = ps.data.numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()


def accuracy(model_path):
    batch_size = 64
    file = 'mnist_only_fully_connected_layer.pt'
    # file = 'model.pt'
    # model = nn.Module()
    # network_state_dict = torch.load(file)
    # model.load_state_dict(network_state_dict)
    model = torch.load(model_path)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_set = datasets.MNIST('MNIST/', download=False, train=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    correct_count, all_count = 0, 0
    for images, labels in test_loader:
        for i in range(len(labels)):


            img = images[i].view(1, 784)

            with torch.no_grad():
                logps = model(img)

            # print(logps)
            ps = torch.exp(logps)
            # print(ps)
            probabilities = list(ps.numpy()[0])
            predicted_label = probabilities.index(max(probabilities))
            true_label = labels.numpy()[i]
            if true_label == predicted_label:
                correct_count += 1
            all_count += 1

    print("Number Of Images Tested =", all_count)
    print("Number Of Images Correct =", correct_count)


    print("\nModel Accuracy =", (correct_count / all_count))


def test_and_show_image(model_path, img):
    model = torch.load(model_path)
    # img = img.view(1, 784)
    with torch.no_grad():
        logpb = model(img)
    # Output of the network are log-probabilities, need to take exponential for probabilities
    pb = torch.exp(logpb)
    probab = list(pb.numpy()[0])
    print("Predicted Digit =", probab.index(max(probab)))
    # classify(img.view(1, 28, 28), pb)
    return probab.index(max(probab)), max(probab)


def test_and_show_image_linear(model_path, img):
    model = torch.load(model_path)
    img = img.view(1, 784)
    with torch.no_grad():
        logpb = model(img)
    # Output of the network are log-probabilities, need to take exponential for probabilities
    pb = torch.exp(logpb)
    probab = list(pb.numpy()[0])
    print("Predicted Digit =", probab.index(max(probab)))
    # classify(img.view(1, 28, 28), pb)
    return probab.index(max(probab)), max(probab)


def imageprepare(image_path):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(image_path)
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels
    #
    # if width > height:  # check which dimension is bigger
    #     # Width is bigger. Width becomes 20 pixels.
    #     nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
    #     if (nheight == 0):  # rare case but minimum is 1 pixel
    #         nheight = 1
    #         # resize and sharpen
    #     img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
    #     wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
    #     newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    # else:
    #     # Height is bigger. Heigth becomes 20 pixels.
    #     nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
    #     if (nwidth == 0):  # rare case but minimum is 1 pixel
    #         nwidth = 1
    #         # resize and sharpen
    #     img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
    #     wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
    #     newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    newImage.paste(im)
    newImage.save("sample.png")

    tv = list(newImage.getdata())  # get pixel values
    print(tv)
    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    # tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    tva = [x * 1.0 for x in tv]
    return tva


def image_to_tensor(filename):
    x = [imageprepare(filename)]  # file path here
    # print(len(x))  # mnist IMAGES are 28x28=784 pixels
    # print(x[0])
    # Now we convert 784 sized 1d array to 24x24 sized 2d array so that we can visualize it
    newArr = [[0 for d in range(28)] for y in range(28)]
    k = 0
    for i in range(28):
        for j in range(28):
            newArr[i][j] = x[0][k]
            k = k + 1

    image = torch.tensor(newArr)
    return image


# explore_data()

# model_path = 'linear_model.pt'
model_path = 'convolutional_model.pt'
# training(model_path)
# accuracy(model_path)

def test():
    for i in range(2000,3000):
        p = transforms.Compose([transforms.Resize((28, 28))])
        path = 'IMAGE/'+str(i)+'.png'
        # path = 'image.png'
        # model = MnistModel(10)
        # model = torch.load(model_path)
        image_fp = open(path, "rb")
        img = Image.open(image_fp).convert('L')
        img = p(img)
        # img.convert('L')
        img = transforms.ToTensor()(img)
        img = img.unsqueeze(0)
        print(img.shape)
        # transforms.Normalize(img, (0.5,), (0.5,))
        # print(img)
        torchvision.utils.save_image(img, 'haha.png')

        # print(model.forward(img))
        test_and_show_image(model_path, img)

        # break

# test()



def training2(model_path):
    # model = MnistModel(10)
    model = torch.load(model_path)
    batch_size = 64
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = datasets.MNIST('MNIST/', download=False, train=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    epochs = 200
    running_lost_list = []
    epochs_list = []

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    for epoch in range(epochs):
        running_lost = 0
        for images, labels in train_loader:
            # images = images.view(images.shape[0], -1)
            model.train()
            # print(images.shape)
            # break
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_lost += loss.item()

        print("Epoch {} - Training Loss: {}".format(epoch, running_lost / len(train_loader)))
        # accuracy2(model_path)
    torch.save(model, 'convolutional_model.pt')


def accuracy2(model_path):
    batch_size = 64
    model = torch.load(model_path)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_set = datasets.MNIST('MNIST/', download=False, train=False, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    correct_count, all_count = 0, 0
    for images, labels in test_loader:
        for i in range(len(labels)):

            torchvision.utils.save_image(images[i], 'image.png')
            p = transforms.Compose([transforms.Resize((28, 28))])
            image_fp = open('image.png', "rb")
            img = Image.open(image_fp).convert('L')
            img = p(img)
            img = transforms.ToTensor()(img)
            img = img.unsqueeze(0)

            # img = images[i].unsqueeze(0)
            with torch.no_grad():
                logps = model(img)

            # print(logps)
            ps = torch.exp(logps)
            # print(ps)
            probabilities = list(ps.numpy()[0])
            predicted_label = probabilities.index(max(probabilities))
            true_label = labels.numpy()[i]
            if true_label == predicted_label:
                correct_count += 1
            all_count += 1

        print("Number Of Images Tested =", all_count)
        print("Number Of Images Correct =", correct_count)
        print("\nModel Accuracy =", (correct_count / all_count))


# training loss = 0.02347 0.01988 0.01719 0.01501 0.0118
# accuracy = 0.8478 0.8235 0.8771 0.8909 0.8954 0.88
# training2('convolutional_model.pt')
accuracy2('convolutional_model.pt')