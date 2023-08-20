from mimetypes import init
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report, f1_score



# Trains the neural network over one epoch and outputs relevent statistics
def train_neural_network(train_loader, epoch, net, device, optimizer, criterion, scheduler, loss_values, train_acc_values):
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    all_labels_train = []
    all_preds_train = []

    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # Move the inputs and labels to the GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
       
        # print statistics
        current_loss = loss.item()
        running_loss += current_loss
        loss_values.append(current_loss)

        # calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        # store all true labels and predicted labels
        all_labels_train.extend(labels.tolist())
        all_preds_train.extend(predicted.tolist())

    # Total accuracy of predictions this epoch
    train_acc = float(100 * correct_train / total_train)

    # Append accuracy after each epoch
    train_acc_values.append(train_acc)

    print('Training set accuracy: %.1f %%' % train_acc, 'Mean loss: %.3f' % (running_loss / (i + 1)))

    # Compute confusion matrix for training set and output statistical info
    print_metrics(all_labels_train, all_preds_train, "Training set")

    return loss_values, train_acc_values, all_labels_train, all_preds_train, total_train


# Runs the validation set through the neural network and outputs relevent statistics about the predictions.
def check_validation_dataset(val_loader, net, device, val_name="Validation set", image_analysis=False):
    correct_val = 0
    total_val = 0
    all_labels_val = []
    all_preds_val = []
    correct_preds = []
    incorrect_preds = []

    # accumulated_results = torch.zeros((10, 10)).to(device)

    for data in val_loader:
        images, labels = data
        # Move the images and labels to the GPU
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        
        _, predicted = torch.max(outputs.data, 1)
        total_val += labels.size(0)
        correct_val += (predicted == labels).sum().item()

        # store all true labels and predicted labels
        all_labels_val.extend(labels.tolist())
        all_preds_val.extend(predicted.tolist())

        # if true, get predictions, correct labels, and prediction probabilities of up to ten different correctly and incorrectly predicted images
        if (image_analysis):
            perform_image_analysis(correct_preds, incorrect_preds, predicted, labels, outputs, images)

        '''
        softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)

        # Perform the multiplication operation
        for output in softmax_outputs:
            result = torch.ger(output, output)  # outer product
            # replace diagonal with original values
            for i in range(result.size(0)):
                result[i, i] = output[i]
            accumulated_results += result  # accumulate the results
        '''

    val_accuracy = float(100 * correct_val / total_val)

    print(f'\n\n{val_name} accuracy: %.1f %%' % val_accuracy)

    # Compute confusion matrix for validation set and output statistical info
    print_metrics(all_labels_val, all_preds_val, val_name)

    total_displacement = get_total_displacement(all_labels_val, all_preds_val)

    # if true, print predictions, correct labels, and prediction probabilities of up to ten different correctly and incorrectly predicted images
    if image_analysis:
        print_image_analysis(correct_preds, incorrect_preds)

    # print(accumulated_results)

    return val_accuracy, total_displacement


def get_total_displacement(all_labels_val, all_preds_val):
    count_true_dict = count_true_labels(all_labels_val)
    count_pred_dict = count_predictions(all_preds_val)

    prediction_difference = 0

    for key in count_true_dict.keys():
        predicted_val = count_pred_dict[key]
        actual_val = count_true_dict[key]
        prediction_difference += int(0.5 * abs(predicted_val - actual_val))

    return prediction_difference


# Get prediction probabilities of up to ten different (belonging to different classes) correctly and incorrectly predicted images
def perform_image_analysis(correct_preds, incorrect_preds, predicted, labels, outputs, images):
    # Calculate softmax probabilities
    probabilities = F.softmax(outputs, dim=1)

    # Find correct and incorrect predictions
    correct = (predicted == labels)
    incorrect = (predicted != labels)

    # Store correct and incorrect predictions of at most one image from each class
    for i in range(len(images)):
        if len(correct_preds) < 10 and correct[i] and (len(correct_preds) == 0 or correct_preds[-1][2] != labels[i]):
            correct_preds.append((images[i], predicted[i], labels[i], probabilities[i]))
        elif len(incorrect_preds) < 10 and incorrect[i] and (len(incorrect_preds) == 0 or incorrect_preds[-1][2] != labels[i]):
            incorrect_preds.append((images[i], predicted[i], labels[i], probabilities[i]))

        # Stop when we have enough predictions
        if len(correct_preds) >= 10 and len(incorrect_preds) >= 10:
            break

    return correct_preds, incorrect_preds


def print_image_analysis(correct_preds, incorrect_preds):
    # Output the predictions
    for i, (image, prediction, label, probabilities) in enumerate(correct_preds + incorrect_preds):
        print(f"Image {i+1}:")
        print(f"  Prediction: {prediction}")
        print(f"  Actual label: {label}")
        print(f"  Probabilities: {(probabilities * 100)}")
        plt.imshow(image.cpu().numpy().transpose((1, 2, 0)))
        plt.show()



def get_mean_squared_error(predicted_accuracy, actual_accuracy):
    return ((1 - (predicted_accuracy / 100)) - (1 - (actual_accuracy / 100))) ** 2


def count_predictions(predicted_labels):
    count_pred_dict = Counter(predicted_labels)
    sorted(count_pred_dict.items(), key=lambda item: item[1])

    return count_pred_dict

def count_true_labels(true_labels):
    count_true_dict = Counter(true_labels)
    sorted(count_true_dict.items(), key=lambda item: item[1])

    return count_true_dict


def print_test_info(batch_size, learning_rate, learning_rate_decay, momentum, validation_size, dataset_expansion_multiplier, 
                    max_images_trained, dyna_sampler):

    print(f"Hyperparameters used are as follows:\n\nbatch size = {batch_size}, learning rate = {learning_rate}, "
          f"learning rate decay = {learning_rate_decay}, momentum = {momentum}\nvalidation size = {validation_size}, "
          f"dataset expansion multiplier = {dataset_expansion_multiplier}, max images trained = {max_images_trained}\n"
          f"loss function = Cross Entropy Loss, optimizer = SGD")
    print(f"\ndynamic weights based on F1 scores (power of {dyna_sampler.power}), minimum dampening = {dyna_sampler.dampening_min}" 
          f", maximum dampening = {dyna_sampler.dampening_max}\ninitial dampening = {dyna_sampler.dampening_initial}, "
          f"forward step size = {dyna_sampler.forward_step_size}, reverse step size = {dyna_sampler.reverse_step_size}\n"
          f"sample size drop threshold for next dampening step: {dyna_sampler.forward_step_threshold}")


def print_metrics(true_labels, predictions, name):
    # Compute confusion matrix for training set and output statistical info
    cm_train = confusion_matrix(true_labels, predictions)

    print(f'\n{name} confusion matrix:\n')
    print(cm_train)
    print(f'\n{name} classification report:\n')
    print(classification_report(true_labels, predictions, digits=3))


def print_simulation_metrics(accuracy, CI_vals, confidence_interval, mean_squared_error, epoch, val_name="Validation set"):
    print(f'{val_name} simulated unlabelled model accuracy: %.1f %%' % accuracy, ' \nSimulated unlabelled model confidence interval '
         '(%.1f %%' % (confidence_interval * 100), ') from %.1f %%' % CI_vals[0], ' to %.1f %%' % CI_vals[1])
    print(f'Mean squared error of current prediction: {mean_squared_error[epoch]}')
    print(f'Mean squared error of all predictions: {(sum(mean_squared_error) / (epoch + 1))}')

