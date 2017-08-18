from distributed_optimizer import DistributedOptimizer


GATE_OP=1

"""
Distributed Optimizer for Synchronous Training
"""


class SyncOptimizer(DistributedOptimizer):

    def __init__(self, 
                 optimizer,
                 communicator, 
                 use_locking=False,
                 name="SyncOpt"):

        DistributedOptimizer.__init__(self, 
                                      optimizer,
                                      communicator)




    def compute_gradients(self, 
                          loss, 
                          var_list=None,
                          gate_gradients=GATE_OP, 
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          grad_loss=None):

           
        gradz, varz = zip(*self._opt.compute_gradients(loss,
                                                       var_list,
                                                       gate_gradients,
                                                       aggregation_method,
                                                       colocate_gradients_with_ops,
                                                       grad_loss))

        
        agg_grads = self._comm.aggregate_tensors(gradz)

        avg_grads = [grad/self._num_workers for grad in
                agg_grads]

        return zip(avg_grads, varz)


    def apply_gradients(self, 
                        grads_and_varz, 
                        global_step=None,
                        name=None):


        return self._opt.apply_gradients(grads_and_varz,
                                         global_step, 
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
                                            var_list,
                                            gate_gradients, 
                                            aggregation_method,
                                            colocate_gradients_with_ops,
                                            grad_loss)

        return self.apply_gradients(grads_varz, global_step)


