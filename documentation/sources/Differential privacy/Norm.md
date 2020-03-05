<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/differential_privacy/norm.py#L5)</span>
## SensitivityNorm class

```python
shfl.differential_privacy.norm.SensitivityNorm()
```


This class defines the interface that must be implemented to compute the sensitivity norm between
two values in a normed space.


---
## SensitivityNorm methods

### compute


```python
compute(x_1, x_2)
```



The compute method receives the result of apply a certain function over private data and
returns the norm of the responses

__Arguments:__

- __x_1__: array response from a concrete query over database 1
- __x_2__: array response from the same query over database 2
    
----

<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/differential_privacy/norm.py#L23)</span>
### L1SensitivityNorm

```python
shfl.differential_privacy.norm.L1SensitivityNorm()
```


Implements the L1 norm of the difference between x_1 and x_2
