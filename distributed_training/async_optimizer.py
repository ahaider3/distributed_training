import tensorflow as tf

from distributed_optimizer import DistributedOptimizer


GATE_OP=1

"""
Distributed Optimizer for Asynchronous Training
"""


class AsyncOptimizer(DistributedOptimizer):

    def __init__(self, 
                 optimizer,
                 communicator, 
                 varz=None,
                 softsync=None,
                 use_locking=False,
                 name="AsyncOpt"):

        with tf.variable_scope(name):

            DistributedOptimizer.__init__(self, 
                                          optimizer,
                                          communicator)

            self._softsync = 1 if softsync is None else softsync

            self._varz = tf.trainable_variables() if varz is None else varz


            self._center = [tf.get_variable("center_%d" % ind,
                                                    initializer=v.initialized_value(),
                                                    trainable=False)
                                for ind, v in enumerate(self._varz)]



    def update_center(self, grads):
        g_grads = self.get_global_grads(grads)
        return self._opt.apply_gradients(zip(g_grads,
            self._center))       



    def get_global_grads(self, grads):


        global_grads = self._comm.aggregate_tensors(grads)
        # TODO simplify this at the cost of performance?
        if self._softsync == 1:
            return global_grads
        else:



            return [g/self._softsync for g in global_grads]

            



    def compute_gradients(self, 
                          loss, 
                          var_list=None,
                          gate_gradients=GATE_OP, 
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          grad_loss=None):

           
        gradz, varz = zip(*self._opt.compute_gradients(loss,
                                                       self._varz,
                                                       gate_gradients,
                                                       aggregation_method,
                                                       colocate_gradients_with_ops,
                                                       grad_loss))
       




        return zip(gradz, varz)



    def apply_gradients(self, 
                        grads_and_varz, 
                        global_step=None,
                        name=None):

        apply_ops = []
        target, _ = zip(*grads_and_varz)

        train_ops = self.update_center(target)


        with tf.control_dependencies([train_ops]):
                assign_ops = [tf.assign(v, c).op
                        for v,c in zip(self._varz, self._center)]

                apply_ops += assign_ops

       
        apply_ops.append(train_ops)

        return apply_ops

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
                                            var_list,
                                            gate_gradients, 
                                            aggregation_method,
                                            colocate_gradients_with_ops,
                                            grad_loss)

        return self.apply_gradients(grads_varz, global_step)


    def get_center(self):
        return self._center


