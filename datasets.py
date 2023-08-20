from torch.utils.data import Dataset, Subset
from torchvision import transforms
from math import floor
from PIL import Image
import numpy as np
import os


# Defines the training, validation, and augmented validation sets. If the training set needs to be expanded using duplicate augmented images,
# _create_indices creates indices for the duplicates (in addition to the originals).
class CustomDataset(Dataset):
    def __init__(self, subset, class_counts, transform=None, target_transform=None, augmentations_per_class=None):
        self.subset = subset
        self.dataset = subset.dataset
        self.indices = subset.indices
        self.transform = transform
        self.target_transform = target_transform
        self.augmentations_per_class = augmentations_per_class if augmentations_per_class is not None else {}
        self.class_counts = class_counts if class_counts is not None else {}
        self.indices = self._create_indices()

    def _create_indices(self):
        indices = []
        next_class_label = 0
        for i in self.indices:
            _, label = self.dataset[i]
            class_label = self.dataset.classes[label]
            augmentations_per_class = self.augmentations_per_class.get(class_label, 0)
            if int(class_label) is next_class_label:
                num_images_in_class = self.class_counts[class_label]
                augmentations_per_image_remainder = augmentations_per_class % num_images_in_class
                remainder_add = 1
                next_class_label += 1
            if augmentations_per_image_remainder <= 0:
                remainder_add = 0
            augmentations_per_image = (augmentations_per_class // num_images_in_class) + remainder_add
            indices.extend([i] * (1 + augmentations_per_image))
            augmentations_per_image_remainder -= 1
        return indices

    def __getitem__(self, index):
        index = self.indices[index]
        image, label = self.dataset[index]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.indices)

# Counts the number of images in each class for the training set, validation set, and combined set.
def get_class_counts(dataset, validation_size, class_labels):
    total_class_counts = {}

    # Counts the number of images in the original dataset
    for _, label in dataset:
        class_label = class_labels[label]
        if class_label not in total_class_counts:
            total_class_counts[class_label] = 0
        total_class_counts[class_label] += 1

    val_class_counts = {}
    train_class_counts = {}

    # Gets the number of images in the training and validation sets.
    for class_label, count in total_class_counts.items():
        if validation_size < 1:
            val_class_counts[class_label] = floor(validation_size * count)
        else:
            val_class_counts[class_label] = validation_size

        train_class_counts[class_label] = count - val_class_counts[class_label]

    return total_class_counts, train_class_counts, val_class_counts


# Creates the training, validation, and augmented validation subsets from the original dataset's indices
def get_subsets(dataset, class_indices, val_class_counts, targets):
    # Randomly select 'val_class_counts' indices from each class for the validation set
    val_indices = np.concatenate([np.random.choice(indices, val_class_counts[i], replace=False) for i, indices in class_indices.items()])

    # The remaining indices will be used for the training set
    train_indices = [i for i in range(len(targets)) if i not in val_indices]

    # Create the training, validation, and the augmented validation sets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    augmented_val_dataset = val_dataset

    return train_dataset, val_dataset, augmented_val_dataset


# Gets the number of duplicate images that should be created (and augmented) for each class in the training dataset
def get_num_of_augmentations(max_class_size, count):
    num_of_augmentations = max_class_size - count
    if (num_of_augmentations > 0):
        return num_of_augmentations
    else:
        return 0

# Saves images in the dataset to the specified folder 
def save_images(dataset, folder):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i in range(len(dataset)):
        # Get the image and its label
        image, label = dataset[i]
        # Convert the PyTorch tensor to a PIL image
        image = transforms.ToPILImage()(image)
        # Define the path where the image will be saved
        path = os.path.join(folder, f"{i}_{label}.png")
        # Save the image
        image.save(path)


# Defines the transformations for data augmentation.
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Flip image horizontally
    transforms.RandomVerticalFlip(),  # Flip image vertically
    transforms.RandomAffine(90, translate=(0.15, 0.15), scale=(0.9, 1.15)),  # Rotate image up to 90 degrees, shift pixels up to 15% in any direction, and rescale from 90% to 115% of original size
    transforms.ColorJitter(brightness=0.25),  # Change brightness by up to 25%
    transforms.ToTensor(),  # Convert the images to PyTorch tensors
])

# Defines transformation without data augmentation.
transform_val = transforms.Compose([
    transforms.ToTensor(),  # Convert the images to PyTorch tensors
])