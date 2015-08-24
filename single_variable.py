#!/usr/bin/env python

from __future__ import division

class LeastSquares(object):
    """
    implements linear regression using ordinary least squares

    training_data must be in form [[x1, y1], [x2, y2]]

    a good explanation: http://www.ismll.uni-hildesheim.de/lehre/ml-07w/skript/ml-2up-01-linearregression.pdf
    """

    def __init__(self, training_data):
        self.training_data = training_data

    def calculate(self):
        """ calculate linear regression formula """

        mean_x = sum([i[0] for i in self.training_data]) / len(self.training_data)
        mean_y = sum([i[1] for i in self.training_data]) / len(self.training_data)

        quotient = 0
        divisor = 0
        for i in self.training_data:
            quotient += (i[0] - mean_x) * (i[1] - mean_y)
            divisor += (i[0] - mean_x) ** 2

        self.m = quotient / divisor
        self.b = mean_y - self.m * mean_x

    def get_formula(self):
        """ return the formula after running gradient descent """

        return "{0}*x + {1}".format(round(self.m, 4), round(self.b, 4))

class GradientDescent(object):
    """
    implements linear regression using gradient descent

    training_data must be in form [[x1, y1], [x2, y2]]
    """

    def __init__(self, training_data, init_m=0, init_b=0, learning_rate=0.001, max_iterations=5000, max_cost_diff=0.005):
        self.training_data = training_data

        # initial value for m in y = m*x + b
        self.init_m = init_m

        # initial value for b in y = m*x + b
        self.init_b = init_b
        self.learning_rate = learning_rate

        # max number of iterations to follow gradient descent before stopping
        self.max_iterations = max_iterations

        # max difference in the cost between descents before stopping
        # **little to no difference indicates convergence
        self.max_cost_diff = max_cost_diff

    def hypothesis(self, m, x, b):
        """ calculate hypothesis function """

        return m * x + b

    def cost(self, m, b):
        """ calculate the cost function for values of m and b """

        total = 0
        for i in self.training_data:
            total += (self.hypothesis(m, i[0], b) - i[1]) ** 2

        return total / (2 * len(self.training_data))

    def calculate(self):
        """ use gradient descent to calculate linear regression formula """

        count = 0
        last_cost = 0

        b = self.init_b
        m = self.init_m

        while True:
            tmp_b = 0
            tmp_m = 0

            for i in self.training_data:
                tmp_b += self.hypothesis(m, i[0], b) - i[1]
                tmp_m += (self.hypothesis(m, i[0], b) - i[1]) * i[0]

            b -= (self.learning_rate * tmp_b) / len(self.training_data)
            m -= (self.learning_rate * tmp_m) / len(self.training_data)

            count += 1
            tmp_cost = self.cost(m, b)

            if abs(tmp_cost - last_cost) <= self.max_cost_diff or count > self.max_iterations:
                self.b = b
                self.m = m
                break
            # if the cost increases, attempt to lower the learning
            # rate in the effort of avoiding a never-converging gradient descent
            elif tmp_cost > last_cost:
                self.learning_rate = self.learning_rate * 0.5
            else:
                last_cost = tmp_cost

    def get_formula(self):
        """ return the formula after running gradient descent """

        return "{0}*x + {1}".format(round(self.m, 4), round(self.b, 4))
