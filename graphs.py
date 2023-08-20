import matplotlib.pyplot as plt
import numpy as np


# Plots the accuracy of the training, validation, and augmented validation sets after each epoch and the loss after each batch.
def create_primary_plots(epoch, batches_per_epoch, loss_values, train_acc_values, val_acc_values, augmented_val_acc_values):

    # Create an array for the x-axis of the loss plot
    x_loss = np.linspace(0, epoch, num=epoch*batches_per_epoch)

    # Plot training and validation accuracy values
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epoch + 1), train_acc_values, label='Train')
    plt.plot(range(1, epoch + 1), val_acc_values, label='Validation')
    plt.plot(range(1, epoch + 1), augmented_val_acc_values, label='Augmented Validation')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss values
    plt.subplot(1, 2, 2)
    plt.plot(x_loss, loss_values)
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

# Plots the predicted accuracy (from the simulated unlabeled model) of validation and augmented validation sets after each epoch, and compares them to the actual accuracies.
def create_unlabeled_set_simulation_plots(epoch, val_acc_values, augmented_val_acc_values, simulated_model_accuracy, simulated_model_CI_lo,
                                          simulated_model_CI_hi, simulated_model_MSE, aug_simulated_model_accuracy, aug_simulated_model_CI_lo, 
                                          aug_simulated_model_CI_hi, aug_simulated_model_MSE):

    # Plot actual validation accuracy values along with predicted values obtained from the simulated model.
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epoch + 1), val_acc_values, label='Validation (actual)')
    plt.plot(range(1, epoch + 1), simulated_model_accuracy, label='Validation (model)')
    plt.plot(range(1, epoch + 1), simulated_model_CI_lo, label='95% Confidence Interval (model)')
    plt.plot(range(1, epoch + 1), simulated_model_CI_hi, label='95% Confidence Interval (model)')
    plt.title('Simulated Accuracy and Confidence Interval')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot actual augmented validation accuracy values along with predicted values obtained from the simulated model.
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epoch + 1), augmented_val_acc_values, label='Augmented Validation (actual)')
    plt.plot(range(1, epoch + 1), aug_simulated_model_accuracy, label='Augmented Validation (model)')
    plt.plot(range(1, epoch + 1), aug_simulated_model_CI_lo, label='95% Confidence Interval (model)')
    plt.plot(range(1, epoch + 1), aug_simulated_model_CI_hi, label='95% Confidence Interval (model)')
    plt.title('Simulated Accuracy and Confidence Interval')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()


    # Plot mean squared error for validation set predictions
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epoch + 1), simulated_model_MSE)
    plt.title('MSE of Val Set Simulated Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')

    # Plot mean squared error for augmented validation set predictions
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epoch + 1), aug_simulated_model_MSE)
    plt.title('MSE of Aug Val Set Simulated Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')




