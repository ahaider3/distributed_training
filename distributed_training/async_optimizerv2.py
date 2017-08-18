import tensorflow as tf

from distributed_optimizer import DistributedOptimizer


GATE_OP=1

"""
Distributed Optimizer for Asynchronous Training
"""


class AsyncOptimizerV2(DistributedOptimizer):

    def __init__(self, 
                 optimizer,
                 communicator, 
                 num_workers,
                 varz=None,
                 tau=1,
                 modulate_lr=0,
                 normalize_gradients=0,
                 softsync=None,
                 use_locking=False,
                 name="AsyncOpt"):

        with tf.variable_scope(name):

            DistributedOptimizer.__init__(self, 
                                          optimizer,
                                          communicator, 
                                          num_workers)

            self._tau = tau
            self._norm_grad = normalize_gradients
            self._modulate_lr = modulate_lr
            self._softsync = int(num_workers) if softsync is None else softsync
#            if self._modulate_lr:
#                self._opts = [tf.train.AdagradOptimizer(.0001/i)  for i in
#                        range(1, self._softsync+1)]
#            else:
#                self._opts = [tf.train.AdagradOptimizer(.0001)] * self._softsync
            with tf.device("/cpu:0"):    
             self._step = tf.get_variable("step", 
                                         initializer=tf.constant_initializer(0),
                                         shape=[],
                                         dtype=tf.int32,
                                         trainable=False)
            self._varz = tf.trainable_variables() if varz is None else varz

            self._accum_grads = [ tf.get_variable("accum_%d" % ind,
                                                  initializer=tf.zeros(shape=tf.shape(v)),
                                                  trainable=False)
                                for ind, v in enumerate(self._varz)]

            num_vars = len(self._varz)
            self._default_grads = [tf.get_variable("default_%d" % ind,
                                                  initializer=tf.zeros(shape=tf.shape(v)),
                                                  trainable=False)
                                for ind, v in enumerate(self._varz)]

            self._center = [tf.get_variable("center_%d" % ind,
                                                    initializer=v.initialized_value(),
                                                    trainable=False)
                                for ind, v in enumerate(self._varz)]
            print(self._center)

    def sync_step(self):
        ops = [self._opt.apply_gradients(
                zip(self._default_grads,self._varz))]

        assigns = [tf.assign(v, c).op
                for v,c in zip(self._varz, self._center)]
        return ops + assigns

    def async_step(self, grads):
        ops = [self._opt.apply_gradients(zip(grads, self._varz))]



        assigns = [v.op for v in self._varz]
        return ops + assigns


    def update_accum(self):
        return [tf.assign(g, tf.zeros(shape=tf.shape(g))).op for
                g in self._accum_grads]


    def keep_accum(self):
        return [g.op for
                g in self._accum_grads]


    def keep_center(self, grads):
        op = self._opt.apply_gradients(zip(grads,
            self._center))
        return op

    def update_center(self, grads):
        g_grads = self.get_global_grads(grads)
        return self._opt.apply_gradients(zip(g_grads,
            self._center))       



    def get_global_grads(self, grads):

        # use gradient normalization
        if self._norm_grad:
            grads = [ g/self._tau for g in grads]

#        global_grads = self._comm.gather_tensors(grads,
#                                                 self._num_workers)
        global_grads = self._comm.aggregate_tensors(grads)
        # TODO simplify this at the cost of performance?
        if self._softsync == self._num_workers:
            return global_grads

        elif self._softsync == 1:
            return [g/self._num_workers for g in global_grads]           

        else:

            assert(self._softsync < self._num_workers)

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
        curr_step = tf.assign_add(self._step, 1)
        target, _ = zip(*grads_and_varz)
        if self._tau > 1:
            curr_grad = [tf.assign_add(ag, g) for ag, g in
                    zip(self._accum_grads, target)]
        else:
            curr_grad = target
        with tf.control_dependencies(curr_grad):
            train_ops = tf.cond(tf.equal(curr_step % self._tau, 0),
                            lambda: self.update_center(curr_grad),
                            lambda: self.keep_center(self._default_grads))

        if self._tau > 1:
            with tf.control_dependencies([train_ops]):
                assign_ops = tf.cond(tf.equal(curr_step % self._tau, 0),
                                lambda: self.update_accum(),
                                lambda: self.keep_accum())

            apply_ops += assign_ops


        with tf.control_dependencies([train_ops]):
            with tf.control_dependencies([train_ops]):
                assign_ops = tf.cond(tf.equal(curr_step % self._tau, 0),
                                lambda: self.sync_step(),
                                lambda: self.async_step(target))

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


