1. This program can take advantage of a GPU to speed it up considerably. Even sped up, though, it can still take 5 hours or longer to complete a proper test of around 40 epochs. Don't be alarmed if you run the code and it doesn't appear to be doing anything for several minutes.
2. Near the top of the main.py file, you'll find most of the hyperparameters. The only things that you can't change from there are the transform parameters, which are at the very bottom of the transforms.py file, and the structure of the neural network (not including the layer widths and kernel sizes, which can be changed from main.py). The structure of the neural net can be modified in neuralnet.py.
3. Before you run the program, you should change the directory names from which to get the images (just below all the hyperparameters).
4. Run the program.
5. Take a nap
6. By the time you wake up, hopefully the test is at least halfway done.