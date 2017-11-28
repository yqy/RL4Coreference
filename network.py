#coding=utf8
from NetworkComponet import *

#theano.config.exception_verbosity="high"
#theano.config.optimizer="fast_compile"

'''
RL for Coref
'''

#### Constants
GPU = True
if GPU:
    print >> sys.stderr,"Trying to run under a GPU. If this is not desired,then modify NetWork.py\n to set the GPU flag to False."
    try:
        theano.config.device = 'gpu'
        print >> sys.stderr,"Use gpu"
    except: pass # it's already set 
    theano.config.floatX = 'float32'
else:
    print >> sys.stderr,"Running with a CPU. If this is not desired,then modify the \n NetWork.py to set\nthe GPU flag to True."
    theano.config.floatX = 'float64'


class NetWork():
    def __init__(self,n_inpt,n_single,n_hidden):
        ## input = 1738 for cn 1374 for en
        ## embedding for each mention = 855 for cn, 673 for en
        ## pair_feature = 28

        activate=ReLU
        #activate=tanh

        dropout_prob = T.fscalar("probability of dropout")

        self.params = []

        self.x_inpt = T.fmatrix("input_pair_embeddings")

        w_h_1,b_h_1 = init_weight(n_inpt,n_hidden,pre="inpt_layer_",special=False,ones=False) 
        self.params += [w_h_1,b_h_1]

        self.hidden_layer_1 = activate(T.dot(self.x_inpt,w_h_1) + b_h_1)
        self.hidden_layer_1_dropout = dropout_from_layer(self.hidden_layer_1,dropout_prob)

        w_h_2,b_h_2 = init_weight(n_hidden,n_hidden/2,pre="hidden_layer_1_",special=False,ones=False) 
        self.params += [w_h_2,b_h_2]

        self.hidden_layer_2_ = activate(T.dot(self.hidden_layer_1,w_h_2) + b_h_2)
        self.hidden_layer_2_dropout_ = dropout_from_layer(activate(T.dot(self.hidden_layer_1_dropout,w_h_2) + b_h_2),dropout_prob)

        w_h_2_,b_h_2_ = init_weight(n_hidden/2,n_hidden/2,pre="hidden_layer_2_",special=False,ones=False) 
        self.params += [w_h_2_,b_h_2_]

        self.hidden_layer_2 = activate(T.dot(self.hidden_layer_2_,w_h_2_) + b_h_2_)
        self.hidden_layer_2_dropout = dropout_from_layer(activate(T.dot(self.hidden_layer_2_dropout_,w_h_2_) + b_h_2_),dropout_prob)

        w_h_3,b_h_3 = init_weight(n_hidden/2,1,pre="output_layer_",special=False,ones=False) 
        self.params += [w_h_3,b_h_3]

        self.output_layer = (T.dot(self.hidden_layer_2,w_h_3) + b_h_3).flatten()
        self.output_layer_dropout = (T.dot(self.hidden_layer_2_dropout,w_h_3) + b_h_3).flatten()
        #self.output_layer = activate(T.dot(self.hidden_layer_2,w_h_3) + b_h_3).flatten()

        ## for single
        self.x_inpt_single = T.fmatrix("input_single_embeddings")

        w_h_1_single,b_h_1_single = init_weight(n_single,n_hidden,pre="inpt_single_layer_",special=False,ones=False) 
        self.params += [w_h_1_single,b_h_1_single]

        self.hidden_layer_1_sinlge = activate(T.dot(self.x_inpt_single,w_h_1_single) + b_h_1_single)
        self.hidden_layer_1_sinlge_dropout = dropout_from_layer(self.hidden_layer_1_sinlge,dropout_prob)

        w_h_2_single,b_h_2_single = init_weight(n_hidden,n_hidden/2,pre="hidden_single_layer_1_",special=False,ones=False) 
        self.params += [w_h_2_single,b_h_2_single]
        self.hidden_layer_2_single_ = activate(T.dot(self.hidden_layer_1_sinlge,w_h_2_single) + b_h_2_single)
        #self.hidden_layer_2_single_ = activate(T.dot(self.hidden_layer_1_sinlge,w_h_2) + b_h_2)
        self.hidden_layer_2_single_dropout_ = dropout_from_layer(activate(T.dot(self.hidden_layer_1_sinlge_dropout,w_h_2_single) + b_h_2_single),dropout_prob)

        w_h_2_single_,b_h_2_single_ = init_weight(n_hidden/2,n_hidden/2,pre="hidden_single_layer_2_",special=False,ones=False) 
        self.params += [w_h_2_single_,b_h_2_single_]
        self.hidden_layer_2_single = activate(T.dot(self.hidden_layer_2_single_,w_h_2_single_) + b_h_2_single_)
        #self.hidden_layer_2_single = activate(T.dot(self.hidden_layer_2_single_,w_h_2_) + b_h_2_)
        self.hidden_layer_2_single_dropout = dropout_from_layer(activate(T.dot(self.hidden_layer_2_single_dropout_,w_h_2_single_) + b_h_2_single_),dropout_prob)

        w_h_3_single,b_h_3_single = init_weight(n_hidden/2,1,pre="output_single_layer_",special=False,ones=False) 
        self.params += [w_h_3_single,b_h_3_single]
        self.output_layer_single = (T.dot(self.hidden_layer_2_single,w_h_3_single) + b_h_3_single).flatten()
        self.output_layer_single_dropout = (T.dot(self.hidden_layer_2_single_dropout,w_h_3_single) + b_h_3_single).flatten()

        self.output_layer_all = T.concatenate((self.output_layer_single,self.output_layer))
        self.output_layer_all_dropout = T.concatenate((self.output_layer_single_dropout,self.output_layer_dropout))

        self.policy = softmax(self.output_layer_all)[0]
        self.policy_dropout = softmax(self.output_layer_all_dropout)[0]

        self.predict = theano.function(
            inputs=[self.x_inpt_single,self.x_inpt],
            outputs=[self.policy],
            #allow_input_downcast=True,
            on_unused_input='warn')

        lr = T.fscalar()
        Reward = T.fscalar("Reward")
        y = T.iscalar('classification')
        ce_lmbda = T.fscalar("cross_entropy_loss")

        l2_norm_squared = sum([(abs(w)).sum() for w in self.params])
        #lmbda_l2 = 0.0000003
        lmbda_l2_train = T.fscalar("Reward")

        self.get_weight_sum = theano.function(inputs=[],outputs=[l2_norm_squared])

        cost = (-Reward) * T.log(self.policy_dropout[y] + 1e-12)\
                + ce_lmbda * T.sum(self.policy_dropout*T.log(self.policy_dropout + 1e-12))\
                + lmbda_l2_train*l2_norm_squared

        grads = T.grad(cost, self.params)
        #grads = [lasagne.updates.norm_constraint(grad, max_norm, range(grad.ndim)) for grad in grads]
        #updates = lasagne.updates.rmsprop(grads, self.params, learning_rate=0.0001)
        clip_grad = 5.0
        cgrads = [T.clip(g,-clip_grad, clip_grad) for g in grads]
        #updates = lasagne.updates.rmsprop(cgrads, self.params, learning_rate=lr)
        updates = lasagne.updates.adadelta(cgrads, self.params, learning_rate=lr)

        self.train_step = theano.function(
            inputs=[self.x_inpt_single,self.x_inpt,y,Reward,lr,lmbda_l2_train,ce_lmbda,dropout_prob],
            outputs=[cost],
            on_unused_input='warn',
            updates=updates)

        self.classification_results = sigmoid(self.output_layer_all)
        self.classification_results_dropout = sigmoid(self.output_layer_all_dropout)
        #self.classification_results = self.policy

        pre_lr = T.fscalar()
        lable = T.ivector()
        lmbda_l2_pretrain = T.fscalar("Reward")

        pre_cost = (- T.sum(T.log(self.classification_results_dropout + 1e-12 )*lable)\
                    - T.sum(T.log(1-self.classification_results+ 1e-12 )*(1-lable)))\
                    + lmbda_l2_pretrain*l2_norm_squared
                    #- T.sum(T.log(1-self.classification_results_dropout+ 1e-12 )*(1-lable)))/(T.sum(lable) + T.sum(1-lable))\

        pregrads = T.grad(pre_cost, self.params)
        pre_cgrads = [T.clip(g,-clip_grad, clip_grad) for g in pregrads]

        #pre_updates = lasagne.updates.rmsprop(pre_cgrads, self.params, learning_rate=pre_lr)
        pre_updates = lasagne.updates.adadelta(pre_cgrads, self.params, learning_rate=pre_lr)

        self.pre_train_step = theano.function(
            inputs=[self.x_inpt_single,self.x_inpt,lable,pre_lr,lmbda_l2_pretrain,dropout_prob],
            outputs=[pre_cost],
            on_unused_input='warn',
            updates=pre_updates)

        # top-pair
        pre_top_lr = T.fscalar()
        lable_top = T.ivector()
        lmbda_l2_pretrain_top = T.fscalar("Reward")

        pre_top_cost = (- T.max(T.log(self.classification_results_dropout + 1e-12 )*lable_top)\
                    - T.min(T.log(1-self.classification_results_dropout+ 1e-12 )*(1-lable_top)))\
                    + lmbda_l2_pretrain_top*l2_norm_squared

        pregrads_top = T.grad(pre_top_cost, self.params)
        pre_top_cgrads = [T.clip(g,-clip_grad, clip_grad) for g in pregrads_top]

        #pre_top_updates = lasagne.updates.rmsprop(pre_top_cgrads, self.params, learning_rate=pre_top_lr)
        pre_top_updates = lasagne.updates.adadelta(pre_top_cgrads, self.params, learning_rate=pre_top_lr)

        self.pre_top_train_step = theano.function(
            inputs=[self.x_inpt_single,self.x_inpt,lable_top,pre_top_lr,lmbda_l2_pretrain_top,dropout_prob],
            outputs=[pre_top_cost],
            on_unused_input='warn',
            updates=pre_top_updates)

        self.pre_predict = theano.function(
            inputs=[self.x_inpt_single,self.x_inpt,lable],
            outputs=[self.classification_results],
            on_unused_input='warn')

        self.score_predict = theano.function(
            inputs=[self.x_inpt_single,self.x_inpt,lable],
            outputs=[self.output_layer_all],
            on_unused_input='warn')


        # cross-entropy
        pre_ce_lr = T.fscalar()
        lable_ce = T.ivector()
        lmbda_l2_pretrain_ce = T.fscalar("Reward")
        pre_ce_lmbda = T.fscalar()

        pre_ce_cost = (- T.sum(T.log(self.policy + 1e-12 )*lable_ce)\
                    - T.sum(T.log(1-self.policy+ 1e-12 )*(1-lable_ce)))/(T.sum(lable_ce) + T.sum(1-lable_ce))\
                    + pre_ce_lmbda * T.sum(self.policy*T.log(self.policy + 1e-12))\
                    + lmbda_l2_pretrain_ce*l2_norm_squared

        pregrads_ce = T.grad(pre_ce_cost, self.params)
        pre_ce_cgrads = [T.clip(g,-clip_grad, clip_grad) for g in pregrads_ce]

        #pre_ce_updates = lasagne.updates.rmsprop(pre_ce_cgrads, self.params, learning_rate=pre_ce_lr)
        pre_ce_updates = lasagne.updates.adadelta(pre_ce_cgrads, self.params, learning_rate=pre_ce_lr)

        self.pre_ce_train_step = theano.function(
            inputs=[self.x_inpt_single,self.x_inpt,lable_ce,pre_ce_lr,lmbda_l2_pretrain_ce,pre_ce_lmbda],
            outputs=[pre_ce_cost],
            on_unused_input='warn',
            updates=pre_ce_updates)

    def show_para(self):
        for para in self.params:
            print >> sys.stderr, para,para.get_value() 

def main():
    r = NetWork(3,4,2)

    zp_x = [[2,3,4],[2,98,4]]
    x_sinlge = [[1,2,3,4]]

    print list(r.predict(x_sinlge,zp_x)[0])
    #print r.predict(zp_x)[0][0]
    #print r.predict(zp_x)[0][1]
    #r.train_step(zp_x,0,0.2,5)
    #r.train_step(zp_x,0,0.2,5)
    #r.train_step(zp_x,0,0.2,5)
    #r.train_step(zp_x,1,1,5)
    #print r.predict(zp_x)[0]

    lable = [0,1,1]
    pre_lr = 5
    l2 = 0.001
    print r.pre_predict(x_sinlge,zp_x,lable)
    #r.show_para()
    print r.pre_top_train_step(x_sinlge,zp_x,lable,pre_lr,l2)
    #print r.pre_train_step(x_sinlge,zp_x,lable,pre_lr,l2)
    #print r.pre_train_step(x_sinlge,zp_x,lable,pre_lr,l2)
    #print r.pre_train_step(x_sinlge,zp_x,lable,pre_lr,l2)
    #print r.pre_predict(x_sinlge,zp_x,lable)
    #r.show_para()

if __name__ == "__main__":
    main()
