from torchvision.datasets import ImageFolder
from torch.utils.data import WeightedRandomSampler, DataLoader
import torch.cuda
import torch.optim as optim
from collections import Counter
from math import ceil
import matplotlib.pyplot as plt
import time

from neuralnet import *
from datasets import *
from train import *
from dynamicsampler import *
from unlabeledaccuracy import *
from graphs import *

# Key neural network hyperparameters (the classics)
validation_size = 0.3   # Set to 0 < x < 1 to use x% of images from each class for the validation set, or x > 1 to use constant x images from each class for the validation set.
dataset_expansion_multiplier = 0.1  # Sets the minimum number of images in the training set from each class to x% of the largest class. If a class has fewer images than the min, augmented duplicates are created.
max_images_trained = 10000000 # If after an epoch, the number of images that were trained on exceeds this number, the program stops training.
batch_size = 256    # Mini batch size                  
learning_rate = 0.03    # Initial learning rate
learning_rate_decay = 0.95  # Set to 0 < x <= 1 to have the learning rate decay (1 - x)% after each epoch (setting it to 1 will cause the learning rate to remain constant).
momentum = 0.9  # Affects how much results from previous gradient descents contribute to the current gradient descent.

# Neural network dimensions
convolution_layer_width = 256   # Width of the convolution layers
kernel1_size = 5    # Kernel size of first convolution layer
kernel2_size = 5    # Kernel size of second convolution layer
linear_layer_width = 2048   # Width of linear layers
output_layer_width = 10     # Width of output layer (number of classes in dataset)

# Dynamic sampling parameters
dynamic_weights_power = 2   # Defines how much each class's f score affects its weights in the next epoch. Set to x >= 0, where x = 1 makes the relationship linear (x = 0 makes the weights static).
dynamic_weights_dampening_min = 0   # The smallest amount of dampening that the dynamic sampler will apply
dynamic_weights_dampening_max = 1   # The largest amount of dampening that the dynamic sampler will apply (this value is reaached asymptotically)
dynamic_weights_dampening_initial = 0.1    # Initial amount of dampening
dynamic_weights_dampening_forward_step_size = 0.1   # How much the dampening increases each time the threshold is reached
dynamic_weights_dampening_reverse_step_size = 0.04  # How much the dampening decays each time the threshold is not reached
dynamic_weights_dampening_step_threshold = 0.9 # If the weight of the largest sampled class falls below this threshold in one epoch, the dampening factor will increase by one step forward.

# Unlabeled dataset simulation parameters
run_unlabeled_set_simulation = False             # Before training the NN, runs the simulation. Set to 'False' if you are using a large validation set (more than one or two thousand).
print_unlabeled_set_simulation_results = False   # Print all accuracies and confidence intervals obtained in simulation.
error_estimate_sample_size = 5000  # Number of random simulations that should be performed
confidence_interval = 0.95         # Used to showcase expected lower and upper bounds of model simulation
probability_weights_offset_min = 1000  # Lower bound for uniform portion of RNG
probability_weights_offset_max = 20000  # Upper bound for uniform portion of RNG
probability_weights_correlation = 0.12   # Governs how much the weights of indices moved from affect the weights of indices moved to (how much precision and recall are correlated)
probability_weights_base_min = 2    # Lower bound of base used for nonuniform portion of RNG
probability_weights_base_max = 2    # Upper bound (inclusive) of base used for nonuniform portion of RNG
probability_weights_power_min = 0   # Lower bound of exponent used for nonuniform portion of RNG
probability_weights_power_max = 19  # Upper bound (inclusive) of exponent used for nonuniform portion of RNG

image_analysis = True   # Set to True to display ten different correctly and incorrectly predicted images as well as the NN's prediction probabilities for each of those images (after the last epoch)
save_images_to_folder = False   # Set to True to save the images in the training, validation, and augmented validation sets to a folder (this will take a looong time, use a smaller set to test transformations)

# Folder from which to get the images
raw_image_folder_name = 'C:/Users/dwypy/source/repos/EARIN_Project/archive/train/EO'

# Folders in which to save resulting training, validation, and augmented validation sets.
train_image_folder_name = 'C:/Users/dwypy/source/repos/EARIN_Project/saved_images/temp_train'
val_image_folder_name = 'C:/Users/dwypy/source/repos/EARIN_Project/saved_images/temp_val'
augmented_val_image_folder_name = 'C:/Users/dwypy/source/repos/EARIN_Project/saved_images/temp_augmented_val'


# Load the dataset
dataset = ImageFolder(raw_image_folder_name)

# Get the class labels
class_labels = dataset.classes

# Get the targets of each image in the dataset
targets = dataset.targets

# Get the indices of images in each class
class_indices = {class_label: np.where(np.array(targets) == i)[0] for i, class_label in enumerate(class_labels)}

# Count the number of images in each class
total_class_counts = {}

# Gets the number of images in the training set, validation set, and combined set.
total_class_counts, train_class_counts, val_class_counts = get_class_counts(dataset, validation_size, class_labels)

# Creates the training, validation, and augmented validation sets.
train_dataset, val_dataset, augmented_val_dataset = get_subsets(dataset, class_indices, val_class_counts, targets)

# Find the size of the largest class
maximum_class_size = int(dataset_expansion_multiplier * max(train_class_counts.values()))

# Calculate the number of augmentations needed for each class
augmentations_per_class = {class_label: get_num_of_augmentations(maximum_class_size, count) for class_label, count in train_class_counts.items()}

# Apply transformations to the data sets
train_dataset = CustomDataset(train_dataset, train_class_counts, transform=transform_train, augmentations_per_class=augmentations_per_class)
val_dataset = CustomDataset(val_dataset, val_class_counts, transform=transform_val)
augmented_val_dataset = CustomDataset(augmented_val_dataset, val_class_counts, transform=transform_train)

# Save the images from the training and validation sets 
if save_images_to_folder:
    save_images(train_dataset, train_image_folder_name)
    save_images(val_dataset, val_image_folder_name)
    save_images(augmented_val_dataset, augmented_val_image_folder_name)


# Calculate the number of images in each class of the training dataset after it has been augmented
new_class_counts = {class_label: count + augmentations_per_class.get(class_label, 0) for class_label, count 
                   in train_class_counts.items()}

train_labels = [train_dataset.dataset.targets[i] for i in train_dataset.indices]


dyna_sampler = DynamicSampler(new_class_counts, train_labels, dynamic_weights_power, dynamic_weights_dampening_min, 
                             dynamic_weights_dampening_max, dynamic_weights_dampening_initial, dynamic_weights_dampening_forward_step_size, 
                             dynamic_weights_dampening_reverse_step_size, dynamic_weights_dampening_step_threshold)


# Print relevent info about current test
print_test_info(batch_size, learning_rate, learning_rate_decay, momentum, validation_size, dataset_expansion_multiplier, max_images_trained, dyna_sampler)


# Runs simulation to estimate accuracy and confidence interval of an unlabeled dataset. The accuracies obtained in the simulation are 
if run_unlabeled_set_simulation:
    sim_probabilities = SimulatedWeights(output_layer_width, error_estimate_sample_size, probability_weights_correlation, probability_weights_offset_min, 
                                        probability_weights_offset_max, probability_weights_base_min, probability_weights_base_max, 
                                        probability_weights_power_min, probability_weights_power_max)

    simulated_unlabeled_mean, simulated_unlabeled_confidence_interval = get_simulated_unlabeled_set_data(sim_probabilities, val_class_counts, confidence_interval, 
                                                                                                         print_unlabeled_set_simulation_results)

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=dyna_sampler.weighted_random_sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
augmented_val_loader = DataLoader(augmented_val_dataset, batch_size=batch_size)

# Instantiate the network
net = Net(convolution_layer_width=convolution_layer_width, kernel1_size=kernel1_size, kernel2_size=kernel2_size, 
          linear_layer_width=linear_layer_width, output_layer_width=output_layer_width)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=learning_rate_decay)

# Print relevent info about neural network used
print(f"\nStructure of neural network:\n\n {net} \n")

# Move the network and loss function to the GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
criterion.to(device)

images_trained = 0
epoch = 0
step = 0
image_analysis_final = False

# Lists to store accuracy and loss values
train_acc_values = []
val_acc_values = []
augmented_val_acc_values = []
loss_values = []
simulated_model_accuracy = []
simulated_model_CI_lo = []
simulated_model_CI_hi = []
simulated_model_MSE = []
aug_simulated_model_accuracy = []
aug_simulated_model_CI_lo = []
aug_simulated_model_CI_hi = []
aug_simulated_model_MSE = []

start_time = time.time()

# loop over the dataset multiple times
while images_trained < max_images_trained:
    print(f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    print('Epoch %d\n\nDampening: %.3f' % ((epoch + 1),  dyna_sampler.current_dampening), ' LR: ', scheduler.get_last_lr())

    # Trains the neural network over one epoch
    loss_values, train_acc_values, all_labels_train, all_preds_train, total_train = train_neural_network(train_loader, epoch, net, device, optimizer, criterion, 
                                                                                                         scheduler, loss_values, train_acc_values)
    
    # Decay Learning Rate
    scheduler.step()

    # Only perform image analysis after last epoch
    if image_analysis and (images_trained + total_train) >= max_images_trained:
        image_analysis_final = True

    # Run validation sets through neural network after each epoch
    with torch.no_grad():
        val_acc, total_displacement = check_validation_dataset(val_loader, net, device, image_analysis=image_analysis_final)
        val_acc_values.append(val_acc)

        if run_unlabeled_set_simulation:
            simulated_model_accuracy.append(simulated_unlabeled_mean[total_displacement])
            simulated_model_CI_lo.append(simulated_unlabeled_confidence_interval[total_displacement][0])
            simulated_model_CI_hi.append(simulated_unlabeled_confidence_interval[total_displacement][1])
            simulated_model_MSE.append(get_mean_squared_error(simulated_unlabeled_mean[total_displacement], val_acc))

            print_simulation_metrics(simulated_unlabeled_mean[total_displacement], simulated_unlabeled_confidence_interval[total_displacement], 
                                     confidence_interval, simulated_model_MSE, epoch)

        augmented_val_acc, total_displacement = check_validation_dataset(augmented_val_loader, net, device, val_name="Augmented validation set", image_analysis=image_analysis_final)
        augmented_val_acc_values.append(augmented_val_acc)
        
        if run_unlabeled_set_simulation:
            aug_simulated_model_accuracy.append(simulated_unlabeled_mean[total_displacement])
            aug_simulated_model_CI_lo.append(simulated_unlabeled_confidence_interval[total_displacement][0])
            aug_simulated_model_CI_hi.append(simulated_unlabeled_confidence_interval[total_displacement][1])
            aug_simulated_model_MSE.append(get_mean_squared_error(simulated_unlabeled_mean[total_displacement], augmented_val_acc))

            print_simulation_metrics(simulated_unlabeled_mean[total_displacement], simulated_unlabeled_confidence_interval[total_displacement], 
                                     confidence_interval, aug_simulated_model_MSE, epoch, val_name="Augmented validation set")

    # Resamples the training set based on the performance of each class after each epoch
    dyna_sampler = dynamic_sampling(dyna_sampler, all_labels_train, all_preds_train)

    # Updates train_loader after resampling
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=dyna_sampler.weighted_random_sampler)
    
    images_trained += total_train
    epoch += 1


print('Finished Training')

end_time = time.time()
total_time = end_time - start_time
print('Total computation time: %.3f seconds' % total_time)

# Calculate the number of mini-batches per epoch
batches_per_epoch = len(train_loader)

# Plots the accuracy of the training, validation, and augmented validation sets as well as the loss values for each epoch.
create_primary_plots(epoch, batches_per_epoch, loss_values, train_acc_values, val_acc_values, augmented_val_acc_values)

# Plots the predicted accuracy (from the simulated unlabeled model) of validation and augmented validation sets after each epoch, and compares them to the actual accuracies.
if run_unlabeled_set_simulation:
    create_unlabeled_set_simulation_plots(epoch, val_acc_values, augmented_val_acc_values, simulated_model_accuracy, simulated_model_CI_lo, 
                                          simulated_model_CI_hi, simulated_model_MSE, aug_simulated_model_accuracy, aug_simulated_model_CI_lo, 
                                          aug_simulated_model_CI_hi, aug_simulated_model_MSE)

plt.tight_layout()
plt.show()