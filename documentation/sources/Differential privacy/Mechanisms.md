<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/differential_privacy/dp_mechanism.py#L5)</span>
### DifferentialPrivacyMechanism

```python
shfl.differential_privacy.dp_mechanism.DifferentialPrivacyMechanism()
```


This is the interface that must be implemented to create an algorithm with the goal to protect
information

----

<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/differential_privacy/dp_mechanism.py#L17)</span>
### UnrandomizedMechanism

```python
shfl.differential_privacy.dp_mechanism.UnrandomizedMechanism()
```


This class doesn't implement randomization mechanism. You might want to send the data without applying
any differential privacy method. Maybe your algorithm is private by design and it is not important that
someone intercepts your data.

----

<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/differential_privacy/dp_mechanism.py#L30)</span>
### RandomizeBinaryProperty

```python
shfl.differential_privacy.dp_mechanism.RandomizeBinaryProperty(prob_head_first=0.5, prob_head_second=0.5)
```


This class uses simple mechanism to add randomness for binary data. This algorithm is described
by Cynthia Dwork and Aaron Roth in their work "The algorithmic Foundations of Differential Privacy".

1.- Flip a coin

2.- If tails, then respond truthfully.

3.- If heads, then flip a second coin and respond "Yes" if heads and "No" if tails.

__Arguments__

- __prob_head_first__: float in [0,1] representing probability to use a random response instead of true value.
    This is equivalent to prob_head of the first coin flip algorithm described by Dwork.
- __prob_head_second__: float in [0,1] representing probability of respond true when random answer is provided.
    Equivalent to prob_head in the second coin flip in the algorithm.

__References__

- [The algorithmic foundations of differential privacy](
   https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
    
----

<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/differential_privacy/dp_mechanism.py#L73)</span>
### LaplaceMechanism

```python
shfl.differential_privacy.dp_mechanism.LaplaceMechanism(sensitivity, epsilon)
```


Implements the laplace mechanism for differential privacy defined by Dwork in their work
"The algorithmic Foundations of Differential Privacy".

Notice that the Laplace mechanism is a randomization algorithm that depends on the sensitivity,
which can be regarded as a numeric query. One can show that this mechanism is
epsilon-differentially private with epsilon = sensitivity/b where b is a constant.

In order to apply this mechanism for a particular value of epsilon, we need to compute
the sensitivity, which might be hard to compute in practice. The framework provides
a method to estimate the sensitivity of a query that maps the private data in a normed space
(see: [SensitivitySampler](../Sensitivity Sampler))

__Arguments:__

- __sensitivity__: float representing sensitivity of the applied query
- __epsilon__: float for the epsilon you want to apply

__References__

- [The algorithmic foundations of differential privacy](
   https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
    