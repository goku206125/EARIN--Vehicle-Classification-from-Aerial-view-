import torch.cuda
import torch
from torch.utils.data import WeightedRandomSampler
from collections import Counter
import math
from sklearn.metrics import f1_score

# The dynamic sampler updates the sampling of each class after every epoch to help classes that are performing poorly catch up and to reduce overfitting
# by classes that are doing too well. The relationship between performance and sampling is adjustable and follows the general formula (performance metric) ^ (power) 
# where 'power' is zero by default, meaning that dynamic sampling simply returns the sampling unchanged by default. If the power is set too high, the neural network
# may start oscillating and potentially diverge. To remedy this, the dynamic sampler also allows for automatic dynamic dampening, which reduces the chance that the
# fuction will be overdampened or underdampened if set manually. If the magnitude of oscillations exceeds the the specified threshold, the dampening factor will
# automatically increase by one step (following the function y = -1/e^x^2). The dampening factor can optionally decrease by one step every time it doesn't
# increase, effectively dampening the dampening.
class DynamicSampler:
    def __init__(self, class_counts, train_labels, power=0, dampening_min=0, dampening_max=0, dampening_initial=0, 
                 forward_step_size=0, reverse_step_size=0, forward_step_threshold=0):
        self.class_counts = class_counts
        self.train_labels = train_labels
        self.total_samples = self.get_total_samples()
        self.starting_weights = self._get_initial_weights()
        self.current_weights = self.starting_weights
        self.image_weights = self.get_image_weights()
        self.weighted_random_sampler = self.get_weighted_random_sampler()
        self.power = power
        self.dampening_min= dampening_min
        self.dampening_max = dampening_max
        self.dampening_initial = dampening_initial
        self.current_step = self._get_initial_step()
        self.current_dampening = self.get_current_dampening()
        self.forward_step_size = forward_step_size
        self.reverse_step_size = reverse_step_size
        self.forward_step_threshold = forward_step_threshold

        
    def _get_initial_step(self):
        return math.sqrt(math.log((self.dampening_min - self.dampening_max) / (self.dampening_initial - self.dampening_max)))

    def _get_initial_weights(self):
        return [self.total_samples / self.class_counts[class_label] for class_label in self.class_counts]


    def step_forward(self):
        self.current_step += self.forward_step_size
        return self.current_step

    def step_reverse(self):
        if self.current_step < self.reverse_step_size:
            self.current_step = 0
        else:
            self.current_step -= self.reverse_step_size
        return self.current_step

    def get_current_dampening(self):
        return (((self.dampening_min - self.dampening_max) / math.exp(self.current_step ** 2)) + self.dampening_max)

    def get_total_samples(self):
        return sum(self.class_counts.values())

    def get_image_weights(self):
        return [self.current_weights[i] for i in self.train_labels]

    def get_weighted_random_sampler(self):
        return torch.utils.data.WeightedRandomSampler(self.image_weights, self.get_total_samples())



# Redistributes the weights of the classes based on the neural network's performance on the training set using f values. 
def dynamic_sampling(dyna_sampler, all_labels_train, all_preds_train):

    f1_values = f1_score(all_labels_train, all_preds_train, average=None)
    f1_multipliers = [1 - f1_value for f1_value in f1_values]

    count_true_dict = Counter(all_labels_train)
    sorted(count_true_dict.items(), key=lambda item: item[1])
    all_labels_dict = count_true_dict

    max_index = max(all_labels_dict, key=all_labels_dict.get)
    max_value = max(all_labels_dict.values())

    # Calculates the weights of the classes by multiplying the f1_multiplier of each class (taken to some power) by its original weight.
    # If dampening is used, the weights from the previous epoch contribute to the new weights. 
    new_class_weights = [(pow(f1_multipliers[int(class_label)], dyna_sampler.power) * dyna_sampler.starting_weights[int(class_label)] * (1 - dyna_sampler.current_dampening)) + 
                         (dyna_sampler.current_weights[int(class_label)] * dyna_sampler.current_dampening) for class_label in dyna_sampler.class_counts]

    # With the exception of the first epoch, if the number of samples of the current epoch's most highly sampled class drops to less than 'dynamic_weights_dampening_step_threshold'%
    # of its sample size in the next epoch's training set, the dynamic sampler steps forward. Otherwise it steps back.
    if dyna_sampler.current_weights != dyna_sampler.starting_weights:
        if (((dyna_sampler.current_weights[max_index] / sum(dyna_sampler.current_weights)) * dyna_sampler.forward_step_threshold) >= (new_class_weights[max_index] / sum(new_class_weights))):
            dyna_sampler.step_forward()
        else:
            dyna_sampler.step_reverse()

    dyna_sampler.current_weights = new_class_weights
    dyna_sampler.image_weights = dyna_sampler.get_image_weights()
    dyna_sampler.current_dampening = dyna_sampler.get_current_dampening()
    dyna_sampler.weighted_random_sampler = dyna_sampler.get_weighted_random_sampler()

    return dyna_sampler