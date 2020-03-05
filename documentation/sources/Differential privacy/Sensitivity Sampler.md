<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/differential_privacy/sensitivity_sampler.py#L6)</span>
### SensitivitySampler

```python
shfl.differential_privacy.sensitivity_sampler.SensitivitySampler()
```


This class implements the algorithm described in the article
Benjamin I. P. Rubinstein and Francesco Aldà. "Pain-Free Random Differential Privacy with Sensitivity Sampling",
accepted into the 34th International Conference on Machine Learning (ICML'2017), May 2017.
It provides a method to estimate the sensitivity of a generic query using a concrete sensitivity norm.

__References__

- [Pain-Free Random Differential Privacy with Sensitivity Sampling](
   https://arxiv.org/pdf/1706.02562.pdf)
    
----

### sample_sensitivity


```python
sample_sensitivity(query, sensitivity_norm, oracle, n, m=None, gamma=None)
```



This method calculates the parameters to sample the oracle and estimates the sensitivity.
One of m or gamma must be provided.

__Arguments__

- __query__: Function to apply over private data (see: [Query](../../Query))
- __sensitivity_norm__: Function to compute the sensitivity norm
    (see: [Norm](../Norm))
- __oracle__: ProbabilityDistribution to sample.
- __n__: int for size of private data
- __m__: int for size of sampling
- __gamma__: float for privacy confidence level

__Returns__

- __sensitivity__: Calculated sensitivity value by the sampler
- __mean__: Mean sensitivity from all samples.
    