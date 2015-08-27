# linear-regression

From some coursework I've been doing, more notes for myself than anything

#### LeastSquares

Implements ordinary least squares for linear regression

```python
training_data = [[1,3], [4,6], [7,10], [8,12], [9,14]]
l = LeastSquares(training_data)
l.calculate()
print l.get_formula()
1.3551*x + 1.1402
```

NormalEquation:

Implements ordinary least squares using the normal equation.
Works with both single and multi variate training sets

```python
training_data = [[1,2,3], [4,5,6], [7,8,10], [8,9,12], [9,11,14]]
l = NormalEquation(training_data)
l.calculate()
print l.get_formula()
0.2333*x0 + 0.0833*x1 + 1.1833*x2 (x0 = 1)
```

#### GradientDescent

Implements gradient descent for linear regression

Single variate:

You choose the intial values for `m` and `b` in the hypothesis function `y = m * x + b` (default is 0,0)

```python
training_data = [[1,3], [4,6], [7,10], [8,12], [9,14]]
l = GradientDescent(training_data)
l.calculate()
print l.get_formula()
1.4669*x + 0.2204

training_data = [[1,3], [4,6], [7,10], [8,12], [9,14]]
l = GradientDescent(training_data, 1, 1)
l.calculate()
print l.get_formula()
1.3515*x + 1.0497
```

Multivariate:

```python
training_data = [[1,2,3], [4,4,6], [7,6,10], [8,7,12], [9,7,14]]
l = GradientDescent(training_data)
l.calculate()
print l.get_formula()
0.146*x0 + 1.0121*x1 + 0.5604*x2 (x0 = 1)

l = GradientDescent(training_data, [0,.5,1])
l.calculate()
print l.get_formula()
0.0127*x0 + 0.6099*x1 + 1.0533*x2 (x0 = 1)
```
