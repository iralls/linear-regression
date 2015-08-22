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

#### GradientDescent

Implements gradient descent for linear regression

You choose the intial values for `m` and `b` in the hypothesis function `y = m * x + b`

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
