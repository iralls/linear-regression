#!/usr/bin/env python

from __future__ import division
import numpy as np

class SingleVariateLeastSquares(object):
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
            x = i[0]
            y = i[1]
            quotient += (x - mean_x) * (y - mean_y)
            divisor += (x - mean_x) ** 2 or 1

        self.m = quotient / divisor
        self.b = mean_y - self.m * mean_x

    def get_formula(self):
        """ return the formula after running gradient descent """

        return "{0}*x + {1}".format(round(self.m, 4), round(self.b, 4))


class SingleVariateGradientDescent(object):
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

        return (m * x) + b

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
                x = i[0]
                y = i[1]
                size = len(self.training_data)
                tmp_b += self.hypothesis(m, x, b) - y
                tmp_m += (self.hypothesis(m, x, b) - y) * x

            b -= (self.learning_rate * tmp_b) / size
            m -= (self.learning_rate * tmp_m) / size

            count += 1
            tmp_cost = self.cost(m, b)

            if abs(tmp_cost - last_cost) <= self.max_cost_diff or count > self.max_iterations:
                self.b = b
                self.m = m
                break
            # if the cost increases, attempt to lower the learning
            # rate in the effort of avoiding a never-converging gradient descent
            elif tmp_cost > last_cost and count > 1000:
                self.learning_rate = self.learning_rate * 0.5
            else:
                last_cost = tmp_cost

    def get_formula(self):
        """ return the formula after running gradient descent """

        return "{0}*x + {1}".format(round(self.m, 4), round(self.b, 4))


class MultiVariateGradientDescent(object):
    """
    implements linear regression using gradient descent

    @params:
        training_data: must be in form [[x1.1,x1.2,...,x1.n,y1], [x2.1,x2.2..,x2.n,y2]]
        ***NOTE - the LAST parameter in each training set will be considered y

        init_parameters: must be in the form [p1,p2,...,pn]
    """

    def __init__(self, training_data, init_parameters=[], learning_rate=0.001, max_iterations=5000, max_cost_diff=0.005):

        # theta0 is treated as 1 for use in equations
        self.x = [[1] + i[:-1] for i in training_data]
        self.y = [i[-1] for i in training_data]

        if len(init_parameters) == 0:
            # initialize 'guess' to 0
            init_parameters = [0 for i in range(len(self.x[0]))]

        # add an initial guess for x0=1
        self.init_parameters = init_parameters
        self.learning_rate = learning_rate

        # max number of iterations to follow gradient descent before stopping
        self.max_iterations = max_iterations

        # max difference in the cost between descents before stopping
        # **little to no difference indicates convergence
        self.max_cost_diff = max_cost_diff

    def hypothesis(self, features, parameters):
        """
        calculate hypothesis function

        features is [x1, x2, ..., xn], where x0 is assumed to be 1
        parameters are integer values [theta0, theta1, ..., thetaN]
        """

        return sum([v * features[k] for k, v in enumerate(parameters)])

    def cost(self, parameters):
        """ calculate the cost function for values in parameters """

        total = 0

        for k,v in enumerate(self.x):
            total += (self.hypothesis(v, parameters) - self.y[k]) ** 2

        return total / (2 * len(self.x))

    def descend(self, original_value, feature_num, parameters):
        """ calcuate gradient descent for a single parameter """

        total = 0

        for i in range(len(self.x)):
            total += (self.hypothesis(parameters, self.x[i]) - self.y[i]) * self.x[i][feature_num]

        return original_value - (self.learning_rate * total) / len(self.x[0])

    def calculate(self):
        """ use gradient descent to calculate linear regression formula """

        count = 0
        last_cost = 0

        parameters = self.init_parameters
        self.final_parameters = parameters

        while True:

            # so we don't overwrite parameters before every
            # theta has been updated
            tmp_parameters = parameters

            for k,v in enumerate(tmp_parameters):
                parameters[k] = self.descend(v, k, tmp_parameters)

            count += 1
            tmp_cost = self.cost(parameters)
            if abs(tmp_cost - last_cost) <= self.max_cost_diff or count > self.max_iterations:
                self.final_parameters = parameters
                break
            # if the cost increases, attempt to lower the learning
            # rate in the effort of avoiding a never-converging gradient descent
            elif tmp_cost > last_cost and count > 1000:
                self.learning_rate = self.learning_rate * 0.5
            else:
                last_cost = tmp_cost

    def get_formula(self):
        """ return the formula after running gradient descent """

        return " + ".join(["{0}*x{1}".format(round(v, 4), k) for k,v in enumerate(self.final_parameters)]) + " (x0 = 1)"


class NormalEquation(object):
    """
    implements linear regression using least squares via the normal equation

    @params:
        training_data: must be in form [[x1.1,x1.2,...,x1.n,y1], [x2.1,x2.2..,x2.n,y2]]
        ***NOTE - the FIRST parameter in each training set will be considered theta0
        meaning it is the 'y-intercept' of the equation
        ***NOTE - the LAST parameter in each training set will be considered y

        init_parameters: must be in the form [p1,p2,...,pn]
    """

    def __init__(self, training_data):

        # theta0 is treated as 1 for use in equations as 'y-intercept'
        self.x = [[1] + i[:-1] for i in training_data]
        self.y = [i[-1:] for i in training_data]

    def calculate(self):
        """ use normal equation to calculate linear regression formula """

        # create X for (X^T * X)^-1 * y
        x = np.matrix(self.x)
        y = np.matrix(self.y)
        self.final_parameters = ((x.transpose() * x).getI() * x.transpose()) * y

    def get_formula(self):
        """ return the formula after running linear regression """

        return " + ".join(["{0}*x{1}".format(round(v[0], 4), k) for k,v in enumerate(self.final_parameters.tolist())]) + " (x0 = 1)"
