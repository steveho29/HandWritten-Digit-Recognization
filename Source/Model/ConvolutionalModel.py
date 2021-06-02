from init import *
from Visualize_Functions import draw_confusion_matrix, draw_loss

MODEL = nn.Sequential(
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

    nn.Flatten(1, -1),
    nn.Linear(128 * 3 * 3, 32),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(32, 10),
    nn.LogSoftmax(dim=1)
)


def get_data(batch_size, is_training=True):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    if is_training:
        train_set = datasets.MNIST('MNIST/', download=False, train=True, transform=transform)
        train_set, val_set = torch.utils.data.random_split(train_set, [50000, 10000])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

        return train_loader, valid_loader
    else:
        test_set = datasets.MNIST('MNIST/', download=False, train=False, transform=transform)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

        return test_loader


def get_accuracy(model, data):
    correct, total = 0, 0
    model.eval()
    for images, labels in data:
        for i in range(len(labels)):
            img = images[i].unsqueeze(0)
            with torch.no_grad():
                output = model(img)
            probabilities = list(torch.exp(output).numpy()[0])
            predicted_label = probabilities.index(max(probabilities))
            true_label = labels.numpy()[i]
            if predicted_label == true_label:
                correct += 1
            total += 1
    return correct / total


def training(n_epochs, batch_size, model_path, is_continue_train=False):
    if not is_continue_train:
        model = MODEL
    else:
        model = torch.load(model_path)

    train_data, validation_data = get_data(batch_size=batch_size, is_training=True)
    # test_data = get_data(batch_size=batch_size, is_training=False)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    best_model = get_accuracy(model, validation_data)

    train_losses = []
    valid_losses = []
    for epoch in range(n_epochs):
        running_loss = 0
        print('Best Model Accuracy: {}'.format(best_model))
        model.train()
        for images, labels in train_data:
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_losses.append(running_loss / len(train_data))
        validation_accuracy = get_accuracy(model, validation_data)
        valid_losses.append(1 - validation_accuracy)

        print("Epoch {} - Training Loss: {} - Validation: {}".format(epoch, running_loss / len(train_data),
                                                                     validation_accuracy))

        if validation_accuracy > best_model:
            print('Save {}'.format(validation_accuracy))
            torch.save(model, model_path)
            best_model = validation_accuracy
        else:
            print('NOT SAVE {}'.format(validation_accuracy))

    draw_loss(valid_losses, train_losses)


def testing_convolutional_model(model_path, batch_size):
    model = torch.load(model_path)
    test_data = get_data(batch_size=batch_size, is_training=False)
    correct, total = 0, 0
    confusion_matrix = np.zeros((10, 10))
    true_labels = []
    predict_labels = []
    for images, labels in test_data:
        for i in range(len(labels)):
            img = images[i].unsqueeze(0)

            # img = images[i].view(1, 784) Using for fully connected model

            with torch.no_grad():
                output = model(img)
            probabilities = list(torch.exp(output).numpy()[0])
            predicted_label = probabilities.index(max(probabilities))
            true_label = labels.numpy()[i]
            if predicted_label == true_label:
                correct += 1
            total += 1
            true_labels.append(true_label)
            predict_labels.append(predicted_label)
            probabilities.append(max(probabilities))
            confusion_matrix[true_label, predicted_label] += 1

    print("Number Of Images Tested =", total)
    print("Number Of Images Correct =", correct)

    print('Accuracy: {}'.format(accuracy_score(true_labels, predict_labels)), end='\n')
    print('Precision: {}'.format(precision_score(true_labels, predict_labels, average='weighted', zero_division=1)),
          end='\n')
    print('Recall: {}'.format(recall_score(true_labels, predict_labels, average='weighted', zero_division=1)), end='\n')
    print('F1 Score: {}'.format(f1_score(true_labels, predict_labels, average='weighted', zero_division=1)), end='\n')
    print('Cohens Kappa: {}'.format(cohen_kappa_score(true_labels, predict_labels)), end='\n')

    draw_confusion_matrix(confusion_matrix)


def run():
    model_path = 'convolutional_model.pt'
    isContinueTraining = True
    training(n_epochs=2, batch_size=64, model_path=model_path, is_continue_train=isContinueTraining)
    testing_convolutional_model(model_path=model_path, batch_size=64)


if __name__ == "__main__":
    run()


