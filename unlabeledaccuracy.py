from operator import add
import numpy as np
import random
import math
from math import floor, ceil
import statistics

# Holds the weights of the probabilities for each sample in the unlabeled dataset simulation. If arguments are not provided (aside from outputs and sample_size),
# the probability weights will be those of a standard random distribution.
class SimulatedWeights:
    def __init__(self, outputs, sample_size, correlation=0, offset_min=0, offset_max=0, base_min=0, base_max=0, power_min=0, power_max=0):
        self.outputs = outputs
        self.sample_size = sample_size
        self.correlation = correlation
        self.offset_min = offset_min
        self.offset_max = offset_max
        self.base_min = base_min
        self.base_max = base_max
        self.power_min = power_min
        self.power_max = power_max
        self.move_from_index_weights = self.set_move_from_index_weights()
        self.move_to_index_weights = self.set_move_to_index_weights()

    def set_move_from_index_weights(self):
        return [[int(self._set_own_weights()) for _ in range(self.outputs)] for _ in range(self.sample_size)]

    def set_move_to_index_weights(self):
        return [[int((self._set_own_weights() * (1 - self.correlation)) + (move_from_index_weight * self.correlation)) 
                 for move_from_index_weight in self.move_from_index_weights[sample_index]] for sample_index in range(self.sample_size)]

    def _set_own_weights(self):
        if self.base_min is self.base_max and self.power_min is self.power_max:
            return random.randrange(self.offset_min, self.offset_max + 1) + pow(self.base_min, self.power_min)
        elif self.base_min is self.base_max:
            return random.randrange(self.offset_min, self.offset_max + 1) + pow(self.base_min, random.randrange(self.power_min, self.power_max + 1))
        elif self.power_min is self.power_max:
            return random.randrange(self.offset_min, self.offset_max + 1) + pow(random.randrange(self.base_min, self.base_max + 1), self.power_min)
        else:
            return random.randrange(self.offset_min, self.offset_max + 1) + pow(random.randrange(self.base_min, self.base_max + 1), 
                                                                                random.randrange(self.power_min, self.power_max + 1))



# Performs a simulation where random errors are purposely introduced to dummy predicted data sets (both labeled and balanced), which initially have 100% accuracy.
# The difference between the number of predicted samples from each class and the correct number of samples in each class increases as more errors are made.
# The results of the simulation are used to estimate the accuracy and confidence interval for all class count displacement values.
# Since the NN will not make purely random errors and will almost certainly perform better with some classes and noticeaby worse with others, the random guesses
# are weighed with exponential functions. Different weights are used when choosing an index to move a prediction from and when choosing an index to move a prediction
# to a new destination, even for the same class. However, outgoing and incoming weight values for the same class are shared to an extent, to simulate the overlapping
# of datasets, where class A is more likely to incorrectly be categorized as class B and vice versa.   
def get_simulated_unlabeled_set_data(sim_probabilities, val_class_counts, confidence_interval, print_unlabeled_set_simulation_results=False):
    
    class_indices, number_of_outputs, validation_size, max_displacement = get_val_set_info(val_class_counts)

    print_simulation_parameters(sim_probabilities, confidence_interval)

    # Performs the simulation and saves every state.
    displacement_distribution = perform_unlabeled_set_simulation(sim_probabilities, val_class_counts)

    # Uses the saved states to extract the mean accuracy and confidence intervals at each total displacement value
    mean, confidence_interval_vals = get_simulation_statistics(displacement_distribution, val_class_counts, confidence_interval, print_unlabeled_set_simulation_results)

    return mean, confidence_interval_vals


def print_simulation_parameters(sim_probabilities, confidence_interval):

    print(f'\nsimulation size = {sim_probabilities.sample_size}, confidence interval used = %.1f %%' % (confidence_interval * 100), 
          f'probability weights correlation = {sim_probabilities.correlation}') 
    print(f'probability weights offset = {sim_probabilities.offset_min} to {sim_probabilities.offset_max}, '
          f'probability weights base = {sim_probabilities.base_min} to {sim_probabilities.base_max}')
    print(f'probability weights power = {sim_probabilities.power_min} to {sim_probabilities.power_max}')
    print(f'probability weights for recall = rng(base) ^ rng(power) + rng(offset)')
    print(f'probability weights for precision = (1 - correlation) * (rng(base) ^ rng(power) + rng(offset)) + (correlation * recall)\n')


def get_val_set_info(val_class_counts):

    number_of_outputs = len(val_class_counts)
    class_indices = [int(index) for index in val_class_counts.keys()]
    validation_size = sum(val_class_counts.values())
    max_displacement = validation_size - sorted(val_class_counts.items(), key=lambda item: item[1])[0][1]

    return class_indices, number_of_outputs, validation_size, max_displacement


# The simulation starts with all predictions being correct. One by one correct, predictions are recategorized randomly into incorrect classes
# until every prediction is incorrect.
def perform_unlabeled_set_simulation(sim_probabilities, val_class_counts):

    class_indices, number_of_outputs, validation_size, max_displacement = get_val_set_info(val_class_counts)

    displacement_distribution = [[0 for _ in range(validation_size + 1)] for _ in range(max_displacement + 1)]
    displacement_distribution[0][0] = sim_probabilities.sample_size

    correct_guesses = [[class_count for class_count in val_class_counts.values()] for _ in range(sim_probabilities.sample_size)]
    wrong_guesses = [[0 for _ in range(number_of_outputs)] for _ in range(sim_probabilities.sample_size)]


    for mistake_number in range(validation_size):
        for k in range(sim_probabilities.sample_size):

            temp_array = sim_probabilities.move_from_index_weights[k].copy()

            while True:
                index_old = random.choices(class_indices, weights=temp_array)[0]
                # The randomly chosen class must still have valid predictions remaining or it must be chosen again.
                if correct_guesses[k][index_old]:
                    correct_guesses[k][index_old] -= 1
                    break
                else:
                    temp_array[index_old] = 0
                    continue
            
            temp_val = sim_probabilities.move_to_index_weights[k][index_old]
            sim_probabilities.move_to_index_weights[k][index_old] = 0

            while True:
                index_new = random.choices(class_indices, weights=sim_probabilities.move_to_index_weights[k])[0]
                # The destination class must be different than the class that was chosen to move.
                if (index_new != index_old):
                    wrong_guesses[k][index_new] += 1
                    break
                else:
                    continue

            sim_probabilities.move_to_index_weights[k][index_old] = temp_val

            # The total guesses for each category consist of all the correct guesses plus all the incorrect ones.
            temp_list = list(map(add, correct_guesses[k], wrong_guesses[k]))
            prediction_difference = 0
            for i, class_count in enumerate(val_class_counts.values()):
                # The difference between the number of predicted samples in a class and the correct amount is halved so it can
                # be used as an index. This number will always be even, so dividing by two won't cause issues.
                prediction_difference += int(0.5 * abs(temp_list[i] - class_count))

            displacement_distribution[prediction_difference][mistake_number + 1] += 1

    return displacement_distribution


# This goes through every state of the simulation to get the mean and confidence interval.
def get_simulation_statistics(displacement_distribution, val_class_counts, confidence_interval, show_simulation_results):

    class_indices, number_of_outputs, validation_size, max_displacement = get_val_set_info(val_class_counts)

    # This is the total number of states that exist for a given displacement value. These include all states, not just
    # final states
    displacement_sums = [sum(displacement_distribution[displacement]) for displacement in range(max_displacement + 1)]

    # These are used to identify the number of values that should be seen on either end before the confidence interval boundary
    # is found. confidence_interval_ceil counts and confidence_interval_floor_counts use the floor and ceil of the calculated
    # values, respectively, so that the actual confidence interval found is never smaller than what was specified.
    confidence_interval_ceil_counts = [max(int(floor(((1 - confidence_interval) / 2) * displace_sum)), 1) * min(displace_sum, 1) 
                                       for displace_sum in displacement_sums]
    confidence_interval_floor_counts = [int(ceil((1 - ((1 - confidence_interval) / 2)) * displace_sum)) for displace_sum in displacement_sums]
    confidence_interval_vals = [[0 for _ in range(2)] for _ in range(max_displacement + 1)]
    mean = [0 for _ in range(max_displacement + 1)]
    
    # Each displacement value will have a mean and confidence interval associated with it.
    for displacement in range(max_displacement + 1):
        ci_sum = 0
        mean_sum = 0
        ci_floor_set = True
        ci_ceil_set = True
        # Goes through every state of every displacement value to find edges of the confidence interval and calculate the
        # expectation values.
        for i, mistake_number_vals in enumerate(displacement_distribution[displacement]):
            ci_sum += mistake_number_vals
            if ci_sum >= confidence_interval_ceil_counts[displacement] and ci_ceil_set:
                confidence_interval_vals[displacement][1] = (1 - float(i / validation_size)) * 100
                ci_ceil_set = False

            if ci_sum >= confidence_interval_floor_counts[displacement] and ci_floor_set:
                confidence_interval_vals[displacement][0] = (1 - float(i / validation_size)) * 100
                ci_floor_set = False

            mean_sum += mistake_number_vals * i

        if displacement_sums[displacement]:
            mean[displacement] = (1 - float(mean_sum / (displacement_sums[displacement] * validation_size))) * 100
        
        # Prints the mean and confidence intervals of every displacement value.
        if show_simulation_results:
            print(f'For displacement of {displacement}, the estimated accuracy is %.1f %%' % mean[displacement], 
                  ' and the %.1f %%' % (confidence_interval * 100), ' confidence interval ranges from '
                  ' %.1f %%' % confidence_interval_vals[displacement][0], ' to %.1f %%' % confidence_interval_vals[displacement][1])


    return mean, confidence_interval_vals

