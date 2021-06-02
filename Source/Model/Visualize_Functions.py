from init import*

def draw_confusion_matrix(confusion_matrix):
    plt.figure(figsize=(15, 10))
    plt.title('Confusion Matrix')
    class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    df_cm = DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
    heat_map = heatmap(df_cm, annot=True, fmt="d")

    heat_map.yaxis.set_ticklabels(heat_map.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=30)
    heat_map.xaxis.set_ticklabels(heat_map.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=30)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def draw_loss(valid_loss, train_loss):
    # print(valid_loss)
    # print(train_loss)
    plt.title("Convolutional Model Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss axis")
    epoch = range(1, len(valid_loss) + 1)

    plt.plot(epoch, train_loss, label="Training Loss")
    plt.plot(epoch, valid_loss, label="Validation Loss")

    plt.legend()
    plt.show()


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


def matplot_classify(img, ps):
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
