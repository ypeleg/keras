from keras.optimizers import SGD, Adagrad, RMSprop, Adadelta, Adam, Adamax
import keras.backend as K

# Forked from: https://gist.github.com/codekansas/6eb99e2d7a2ca3ea174f4ca11b62fe20 
SIGMA = 0.05
  
def noisy_gradient(gradient, noise):
    return [g * K.random_normal(shape=K.shape(g), mean=1.0, std=noise) for g in gradient]
 
 
class noisySGD(SGD):
    def __init__(self, noise=SIGMA, **kwargs):
        self.noise = noise
        super(noisySGD, self).__init__(**kwargs)
 
    def get_gradients(self, loss, params):
        grads = super(noisySGD, self).get_gradients(loss, params)
        return noisy_gradient(grads, self.noise)
 
 
class noisyRMSprop(RMSprop):
    def __init__(self, noise=SIGMA, **kwargs):
        self.noise = noise
        super(noisyRMSprop, self).__init__(**kwargs)
 
    def get_gradients(self, loss, params):
        grads = super(noisyRMSprop, self).get_gradients(loss, params)
        return noisy_gradient(grads, self.noise)
 
 
class noisyAdagrad(Adagrad):
    def __init__(self, noise=SIGMA, **kwargs):
        self.noise = noise
        super(noisyAdagrad, self).__init__(**kwargs)
 
    def get_gradients(self, loss, params):
        grads = super(noisyAdagrad, self).get_gradients(loss, params)
        return noisy_gradient(grads, self.noise)
 
 
class noisyAdadelta(Adadelta):
    def __init__(self, noise=SIGMA, **kwargs):
        self.noise = noise
        super(noisyAdadelta, self).__init__(**kwargs)
 
    def get_gradients(self, loss, params):
        grads = super(noisyAdadelta, self).get_gradients(loss, params)
        return noisy_gradient(grads, self.noise)
 
 
class noisyAdam(Adam):
    def __init__(self, noise=SIGMA, **kwargs):
        self.noise = noise
        super(noisyAdam, self).__init__(**kwargs)
 
    def get_gradients(self, loss, params):
        grads = super(noisyAdam, self).get_gradients(loss, params)
        return noisy_gradient(grads, self.noise)
 
 
class noisyAdamax(Adamax):
    def __init__(self, noise=SIGMA, **kwargs):
        self.noise = noise
        super(noisyAdamax, self).__init__(**kwargs)
 
    def get_gradients(self, loss, params):
        grads = super(noisyAdamax, self).get_gradients(loss, params)
        return noisy_gradient(grads, self.noise)
