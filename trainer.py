import torch
import torch.nn.functional as F


def train(model, train_data, train_label, batch_size, optimizer, epoch, logger, device):
    """
    Training the model, using the VGG16 generated features maps of the images.
    Args:
    -----------
        model: Model architecture to be used to train the model.
        device: Type of device (GPU/CPU) to be used while training model.
        train_data: The feature maps of Training Images
        train_label: The labels of Training Images
        batch_size: The batch size to be used while training the model.
        optimizer: Defining which optimizer to be used.
        epoch: Current Epoch.
        logger: Logger to be used while training the model.
    Returns:
    --------
        train_loss: The loss observed while training the model.
        accuracy: Accuracy of the model over the current training data.
    """
    batch_train_loss = []
    train_loss = 0
    correct_count = 0
    for i in range(train_data[::batch_size,...].shape[0]):

        optimizer.zero_grad() # Setting optimizer value to zero to avoid accumulation of gradient values
        data = torch.from_numpy(train_data[i*batch_size:i*batch_size+batch_size,...]).to(device)
        target = torch.from_numpy(train_label[i*batch_size:i*batch_size+batch_size,...]).to(device)

        out = model(data)
        batch_loss = F.binary_cross_entropy(out.squeeze(), target)

        batch_loss.backward()
        optimizer.step()

        batch_train_loss.append(batch_loss.item())
        train_loss += batch_loss.item()

        pred = torch.round(out.squeeze())
        correct_count += pred.eq(target).sum().item() # Equating Predicted and Label Tensors at each Index value
    
    train_loss /= train_label.shape[0]
    accuracy = 100. * correct_count / train_label.shape[0]

    logger.info(f'Train Loss for Epoch{epoch}: {train_loss}')
    logger.info(f'Train Accuracy for Epoch{epoch}: {accuracy}')

    return train_loss, accuracy


def test(model, test_data, test_label, batch_size, epoch, logger, device):
    """
    """
    model.eval()
    correct_count = 0
    test_loss = 0
    confusion_matrix = torch.zeros(2, 2, dtype=torch.long) #2 Classes
    with torch.no_grad():  # Setting torch to not involve any kind of gradient calculation
        for i in range(test_data[::batch_size,...].shape[0]):

            data = torch.from_numpy(test_data[i*batch_size:i*batch_size+batch_size,...]).to(device)
            target = torch.from_numpy(test_label[i*batch_size:i*batch_size+batch_size,...]).to(device)

            out = model(data)

            test_loss += F.binary_cross_entropy(out.squeeze(), target).item()
            pred = torch.round(out.squeeze()).to(torch.int32)
            target = target.to(torch.int32)
            correct_count += pred.eq(target).sum().item()  # Equating Predicted and Label Tensors at each Index value
            
            for l,p in zip(target, pred):
                confusion_matrix[l, p] += 1

    test_loss /= test_label.shape[0]
    accuracy = 100. * correct_count / test_label.shape[0]
    cls_acc = (confusion_matrix.diag()/confusion_matrix.sum(1))*100
    cat_acc , dog_acc = cls_acc

    logger.info(f'Test Loss for Epoch{epoch}: {test_loss}')
    logger.info(f'Test Accuracy for Epoch{epoch}: {accuracy}')
    logger.info(f'Accuracy for Cat at Epoch{epoch}: {cat_acc}')
    logger.info(f'Accuracy for Dog at Epoch{epoch}: {dog_acc}')

    return test_loss, accuracy, (cat_acc.item(), dog_acc.item())

