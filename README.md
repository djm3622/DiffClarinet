# DiffClarinet

Currently working on training our own version of DDSP for the Karplus strong model with an all-pass filter (slightly different than the tutorial but the same idea).

## Data

The data/ is samples collected from matlab.

The data/ has two datasets currently. One that has a fixed delay with varying L rannging from 40:200 and fixed L with varying delay ranging from 0.999:0.00001:0.99991. They both have an all pass filter with a fixed a = 0.1. The fixed L is 200. The fixed delay is 0.99991. 

Their excitations are a uniform random sample over the range -1 to 1 with length L. After L time the impulse is zeros. 

## Models

The model/ is where the models live. Also optimizers, losses, any anything else necessary to the optimization task.

1. The first model, FindGain, tries to learn a single sample. Given the L and A, try to find the delay gain.
2. The second model, VaryingGain, tries to learn the gain as a function of the input. It again is given L and A, but learns a transformation of input to select the gain, as to learn the gain for many different samples with not the same gain.
3. The third model, FindL, again tries to learn a single sample. This is given delay gain and A and it wants to learn L. Although there will be a problem with this as L must be an integer but the differentian is continous.

## Scripts

The scripts/ are training, inference, and misc.

# Updates

1. We could use Hard Gumbel-Softmax such as from wave2vec 2.0 to select a one hot vector but remain differentiable. This model might come later so lets make it a later test.