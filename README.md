# DiffClarinet

Currently working on training our own version of DDSP for the Karplus strong model with an all-pass filter (slightly different than the tutorial but the same idea).

After extending the mulit-karplus strong model fulling to L, a, and K we will make the clarinet model.

## Data

The data/ is samples collected from matlab.

The data/ has two datasets currently. One that has a fixed delay with varying L rannging from 40:200 and fixed L with varying delay ranging from 0.999:0.00001:0.99991. They both have an all pass filter with a fixed a = 0.1. The fixed L is 200. The fixed delay is 0.99991. 

Their excitations are a uniform random sample over the range -1 to 1 with length L. After L time the impulse is zeros. 

## Models

The model/ is where the models live. Also optimizers, losses, any anything else necessary to the optimization task.

1. The first model, FindGain, tries to learn a single sample. Given the L and A, try to find the delay gain.
2. The second model, VaryingGain, tries to learn the gain as a function of the input. It again is given L and A, but learns a transformation of input to select the gain, as to learn the gain for many different samples with not the same gain.
3. The third model, FindL, again tries to learn a single sample. This is given delay gain and A and it wants to learn L. Although there will be a problem with this as L must be an integer but the differentian is continous.

### Delay-length methods

`KarplusStrongFixed` contains the shared physical synthesizer. The delay-length
estimators in `model/kps/delay_methods.py` extend it with one of four methods:

- `KarplusStrongReinforce`: categorical sampling with REINFORCE.
- `KarplusStrongGumbelSoftmax`: hard Gumbel--Softmax in the forward pass and a
  soft gradient in the backward pass. It draws one categorical sample and
  evaluates one circular response per update.
- `KarplusStrongExhaustive`: evaluates every candidate and selects the minimum
  without training.
- `KarplusStrongPitch`: estimates the period by normalized autocorrelation and
  corrects for the loop filter's group delay.

Select the method with the `delay_method` value near the top of
`scripts/train_single_instance.py`. The comparison fixes the known gain and
all-pass coefficient by default so the result isolates delay selection. Set
`learn_continuous_parameters = True` with REINFORCE or Gumbel--Softmax to
recover the earlier joint-fitting setup. Griffin--Lim is not used for the pitch
baseline because it reconstructs phase from a magnitude spectrogram rather
than estimating pitch, and the dataset already supplies the target waveform.

The method-specific training functions and dispatcher are in
`model/kps/training.py`. Set `delay_method = None` to keep `L` fixed and use the
aligned finite-causal waveform objective. Otherwise, the model subclass selects
its associated REINFORCE, Gumbel--Softmax, exhaustive, or pitch function.

## Scripts

The scripts/ are training, inference, and misc.

# Updates

1. We could use Hard Gumbel-Softmax such as from wave2vec 2.0 to select a one hot vector but remain differentiable. This model might come later so lets make it a later test. Pytorch has a function to do this called, torch.`nn.functional.gumbel_softmax`.
2. Do we need the excitations to match? Test first with no matching excitation, just generated uniform noise in the same way. If this doesn't work try using the exact excitaiton that the matlab model was given.
3. Convert the adaptive models to see if we can learn encoders for each parameter.
4. Next step in comparing pitch estimation and onset detection to the current differentiable algorithm for finding L.
5. Another next step is including the `real` control parameters that a human controls (pluck position, pluck intensity, etc.).
6. Actually implement the clarinet model.
