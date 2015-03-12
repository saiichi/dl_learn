"""
"""


from __future__ import print_function


import numpy
import theano
import theano.tensor as T


class LogisticRegression(object):

    """ Multi-class Logistic Regression Class
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of LR
        """
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
                ),
            name='W',
            borrow=True
            )

        self.b = theano.shared(
            value=numpy.zeros(
                (n_out, ),
                dtype=theano.config.floatX
                ),
            name='b',
            borrow=True
            )

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W), self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]

    def negetive_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.
        """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """
        Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch
        """

        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )

        if y.type.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()