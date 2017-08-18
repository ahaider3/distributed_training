from tensorflow.python.framework import ops
import numpy as np
import tensorflow as tf

from distributed_optimizer import DistributedOptimizer
GATE_OP=1

"""
Optimizer which can wrap any optimizer and 
add on an elastic term
"""


class EAOptimizer(DistributedOptimizer):

    def __init__(self, 
                 optimizer,
                 communicator, 
                 learning_rate, 
                 varz=None, 
                 rho = 16., 
                 beta=None,
                 tau=4,  
                 use_locking=False,
                 name="EASGD"):

        DistributedOptimizer.__init__(self, 
                                      optimizer,
                                      communicator)
        with tf.variable_scope(name):
            varz = tf.trainable_variables() if varz is None else varz

            self._center = [tf.get_variable("center_" + str(i),
                                             initializer=var.initialized_value(),
                                            trainable=False) 
                                            for i, var in enumerate(varz)]


            # TODO this should not be necessary
            with tf.device("/cpu:0"):
                self._step = tf.get_variable("step",
                                             shape=[],
                                             initializer=tf.constant_initializer(0), 
                                             trainable=False,
                                             dtype=tf.int32)

            self._varz = varz

            # set value of tau
            self._tau = tau

        
            # set value of p
            self._n_procs = self._num_workers
        
            # set value of rho
            self._rho = rho


            self._lr = learning_rate
            self._alpha = self._rho * self._lr
 
            # set beta to num_procs * alpha
            self._beta = self._alpha * self._n_procs if beta is None else beta

            self._num_params = float(self.get_num_parameters())


    def update_center(self):

        # get sum of all local var tensors
        sum_vars = self._comm.aggregate_tensors(self._varz)

        # average these values by num_procs
        new_center = [(1 - self._beta) * c + (self._beta * (sv/self._n_procs))
                       for c, sv in zip(self._center, sum_vars)]



        
        assign_ops = []
        for c, new_c in zip(self._center, new_center):
            assign_ops.append(tf.assign(c, new_c))


        return assign_ops

    # get the overall number of parameters
    def get_num_parameters(self):
        shapes = [c.get_shape().as_list() 
                for c in self._center]
        return sum([np.product(shape) for shape in shapes])

    def compute_gradients(self, 
                          loss, 
                          var_list=None,
                          gate_gradients=GATE_OP, 
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          grad_loss=None):

      # get the current masters
        curr_centers = tf.cond(tf.equal(self._step % self._tau, 0),
                               self.update_center,
                               lambda: self._center)

        distance = tf.divide(
                            tf.add_n([tf.reduce_sum(tf.square(v - cv)) 
                                     for v, cv in
                                     zip(self._varz, curr_centers)]),
                            self._num_params)



        # is number of els being calculated each iteration

       
        new_loss = loss + (self._rho/2.) * distance
            
        grads, varz = zip(*self._opt.compute_gradients(new_loss,
                                                       self._varz,
                                                       gate_gradients,
                                                       aggregation_method,
                                                       colocate_gradients_with_ops,
                                                       grad_loss))
        return zip(grads, varz)


    def apply_gradients(self, 
                        grads_and_varz, 
                        global_step=None,
                        name=None):

        tf.assign_add(self._step, 1)

        return self._opt.apply_gradients(grads_and_varz,
                                         self._step, 
                                         name)

    def minimize(self, 
                 loss, 
                 global_step=None, 
                 var_list=None,
                 gate_gradients=GATE_OP, 
                 aggregation_method=None,
                 colocate_gradients_with_ops=False,
                 name=None,
                 grad_loss=None):


        grads_varz = self.compute_gradients(loss, 
                                            self._varz,
                                            gate_gradients, 
                                            aggregation_method,
                                            colocate_gradients_with_ops,
                                            grad_loss)

        return self.apply_gradients(grads_varz, self._step)

    def get_center(self):
        return self._center


