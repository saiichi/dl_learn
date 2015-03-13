"""
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cPickle
import gzip
import os
import sys
import time


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


def load_data(dataset):
    """Loads the dataset
    """
    with gzip.open(dataset) as f:
        train_set, valid_set, test_set = cPickle.load(f)

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=True)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=True)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
                           dataset='mnist.gz', batch_size=600):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model
    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    # Build Model
    print('...building the model.')

    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')    # data
    y = T.ivector('y')    # labels

    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)

    cost = classifier.negetive_log_likelihood(y)
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    valid_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
            y: train_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # Train model
    print('... training the model')
    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience // 2)
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()
    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = [valid_model(i)
                                     for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100
                    )
                )

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss *    \
                      improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    best_validation_loss = this_validation_loss

                    test_losses = [test_model(i)
                                   for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    print(
                        '    epoch %i, minibatch %i/%i, test error of \
                        best model %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100
                        )
                    )

                if patience <= iter:
                    done_looping = True
                    break
    end_time = time.clock
    print(
        'Optimization complete with best validation score of %f %% \
        with test performance %f %%' %
        (
            best_validation_loss * 100,
            test_score * 100
        )
    )
    print('The code run for %d epochs, with %f epochs/sec' %
          (epoch, epoch / (end_time - start_time)))


if __name__ == '__main__':
    sgd_optimization_mnist()
