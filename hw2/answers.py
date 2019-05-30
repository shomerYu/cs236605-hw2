r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    # this params makes the model over-fit in very few steps:
    wstd, lr, reg = 0.01, 0.05, 0.001
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0.01, 0.02, 0.02, 0.001, 0.001
    # ========================
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum,
                lr_rmsprop=lr_rmsprop, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.5
    lr = 0.001
    #  ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""

1. the no drop / drop our comparison is exactly is anticipated, the no drop out overfit the training data and get very 
high train result but very poor test result, on the contrary the dropout networks train performance decreased but the test
results improved. this can by explained by better generality of the model.

2. we can see that the high percentage dropout did not improve the test results, there probably a best drop out value,
and it can by found by trithing the drop value as hyper parameter as well.

"""

part2_q2 = r"""
there is a possibility of the test loss increase while the test accuracy increases as well.
the accuracy of the model dependent only on the comparison between the real class and the estimated class, the cross entropy 
loss is taken into account the mean of the loss function for each class.
for example we can find more classes that are correct (better accuracy) but for the classes that are wrong the loss
function is bigger, resulting in increasing in both accuracy and loss.
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""

1. Base on experiment 1.1 we can conclude that the a shallower network generates higher accuracy
2. The network is not trainable for L in {8, 16}.
    - L=8 we get very deep network (with minimal pooling in comparison to feature expansion).
    This results in model which has a lot of features that we are not able to train with our resources
    - The main reason some networks were able to learn while others are the complexity of the network. for too complex 
    layer the gradients vanish's resulting im very slow learning or no learning at all.


"""

part3_q2 = r"""

In experiment 1.2 we can see that the value of K (number of filters) has a great impact on the tests accuracy
and as expected from the last experiment, the shallower network gave as the best results (L=2, K=256)
Now we can see that the network doesn't fail to train except when we raise L to 8 and thus we can conclude that the failure to train the 
network was given by a ratio L/K too big



"""

part3_q3 = r"""
**Your answer:**
In this experiments the only trainable networks were with L=1,2.
and it achieved a better train accuracy than the previous experiments.


"""


part3_q4 = r"""
1. for the modified network we added some features, we started by adding each con layer batchNorm and dropout layer, after
the training the results weren't good, so we decided to drop some convLayers which gave better results. 

2. the modified network outscored any network in section 1, both at the training results (above 80 percent accuracy and 
test results (above 70% accuracy). in this case all the networks were trainable, better performances could achieve by tring
different hyper parameters.

"""
# ==============
