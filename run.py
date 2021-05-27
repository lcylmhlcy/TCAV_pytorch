import cav as cav
import model as model
import tcav as tcav
import utils as utils
import utils_plot as utils_plot  # utils_plot requires matplotlib
import os
import torch
import activation_generator as act_gen
import tensorflow as tf

# source_dir: where images of concepts, target class and random images (negative samples when learning CAVs) live. Each should be a sub-folder within this directory.
# Note that random image directories can be in any name. In this example, we are using random500_0, random500_1,.. for an arbitrary reason. You need roughly 50-200 images per concept and target class (10-20 pictures also tend to work, but 200 is pretty safe).

# cav_dir: directory to store CAVs (None if you don't want to store)

# target, concept: names of the target class (that you want to investigate) and concepts (strings) - these are folder names in source_dir

# bottlenecks: list of bottleneck names (intermediate layers in your model) that you want to use for TCAV. These names are defined in the model wrapper below.

source_dir = "./image_net_subsets/"

working_dir = './tcav_class_test'
activation_dir = working_dir + '/activations/'
cav_dir = working_dir + '/cavs/'
bottlenecks = ['Mixed_5d', 'Conv2d_2a_3x3']

utils.make_dir_if_not_exists(working_dir)
utils.make_dir_if_not_exists(activation_dir)
utils.make_dir_if_not_exists(cav_dir)

# this is a regularizer penalty parameter for linear classifier to get CAVs.
alphas = [0.1]

target = 'zebra'
concepts = ["dotted", "striped", "zigzagged"]
# random_counterpart = 'random500_1'
LABEL_PATH = './imagenet_comp_graph_label_strings.txt'

mymodel = model.InceptionV3Wrapper(LABEL_PATH)
act_generator = act_gen.ImageActivationGenerator(
    mymodel, source_dir, activation_dir, max_examples=100)

# ---------------------------------------------------------------------------------------------------------------
# num_random_exp: number of experiments to confirm meaningful concept direction. TCAV will search for this many folders named random500_0, random500_1, etc. You can alternatively set the random_concepts keyword to be a list of folders of random concepts. Run at least 10-20 for meaningful tests.

# random_counterpart: as well as the above, you can optionally supply a single folder with random images as the "positive set" for statistical testing. Reduces computation time at the cost of less reliable random TCAV scores.

tf.compat.v1.logging.set_verbosity(0)
num_random_exp = 2  # folders (random500_0, random500_1)

mytcav = tcav.TCAV(target,
                   concepts,
                   bottlenecks,
                   act_generator,
                   alphas,
                   cav_dir=cav_dir,
                   num_random_exp=num_random_exp)
                   
print('Loading mytcav')
results = mytcav.run()
utils_plot.plot_results(results, num_random_exp=num_random_exp)
