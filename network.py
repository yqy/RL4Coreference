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

        w_h_1,b_h_1 = init_weight(n_inpt,n_hidden,pre="inpt_layer_",special=True,ones=False) 
        self.params += [w_h_1,b_h_1]

        self.hidden_layer_1 = activate(T.dot(self.x_inpt,w_h_1) + b_h_1)

        w_h_2,b_h_2 = init_weight(n_hidden,n_hidden/2,pre="hidden_layer_1_",special=True,ones=False) 
        self.params += [w_h_2,b_h_2]

        self.hidden_layer_2_ = activate(T.dot(self.hidden_layer_1,w_h_2) + b_h_2)

        w_h_2_,b_h_2_ = init_weight(n_hidden/2,n_hidden/2,pre="hidden_layer_2_",special=True,ones=False) 
        self.params += [w_h_2_,b_h_2_]

        self.hidden_layer_2 = activate(T.dot(self.hidden_layer_2_,w_h_2_) + b_h_2_)

        w_h_3,b_h_3 = init_weight(n_hidden/2,1,pre="output_layer_",special=True,ones=False) 
        self.params += [w_h_3,b_h_3]

        self.output_layer = (T.dot(self.hidden_layer_2,w_h_3) + b_h_3).flatten()

        ## for single
        self.x_inpt_single = T.fmatrix("input_single_embeddings")

        w_h_1_single,b_h_1_single = init_weight(n_single,n_hidden,pre="inpt_single_layer_",special=True,ones=False) 
        self.params += [w_h_1_single,b_h_1_single]

        self.hidden_layer_1_sinlge = activate(T.dot(self.x_inpt_single,w_h_1_single) + b_h_1_single)

        w_h_2_single,b_h_2_single = init_weight(n_hidden,n_hidden/2,pre="hidden_single_layer_1_",special=True,ones=False) 
        self.params += [w_h_2_single,b_h_2_single]

        self.hidden_layer_2_single_ = activate(T.dot(self.hidden_layer_1_sinlge,w_h_2_single) + b_h_2_single)

        w_h_2_single_,b_h_2_single_ = init_weight(n_hidden/2,n_hidden/2,pre="hidden_single_layer_2_",special=True,ones=False) 
        self.params += [w_h_2_single_,b_h_2_single_]

        self.hidden_layer_2_single = activate(T.dot(self.hidden_layer_2_single_,w_h_2_single_) + b_h_2_single_)

        w_h_3_single,b_h_3_single = init_weight(n_hidden/2,1,pre="output_single_layer_",special=True,ones=False) 
        self.params += [w_h_3_single,b_h_3_single]

        self.output_layer_single = (T.dot(self.hidden_layer_2_single,w_h_3_single) + b_h_3_single).flatten()

        self.output_layer_all = T.concatenate((self.output_layer_single,self.output_layer))

        self.policy = softmax(self.output_layer_all)[0]

        self.predict = theano.function(
            inputs=[self.x_inpt_single,self.x_inpt],
            outputs=[self.policy],
            #allow_input_downcast=True,
            on_unused_input='warn')

        lr = T.fscalar()
        Reward = T.fscalar("Reward")
        y = T.iscalar('classification')

        l2_norm_squared = sum([(abs(w)).sum() for w in self.params])
        lmbda_l2 = 0.0000003

        self.get_weight_sum = theano.function(inputs=[],outputs=[l2_norm_squared])

        cost = (-Reward) * T.log(self.policy[y] + 1e-6)\
                + lmbda_l2*l2_norm_squared

        grads = T.grad(cost, self.params)
        #grads = [lasagne.updates.norm_constraint(grad, max_norm, range(grad.ndim)) for grad in grads]
        #updates = lasagne.updates.rmsprop(grads, self.params, learning_rate=0.0001)
        clip_grad = 5.0
        cgrads = [T.clip(g,-clip_grad, clip_grad) for g in grads]
        updates = lasagne.updates.rmsprop(cgrads, self.params, learning_rate=lr)

        self.train_step = theano.function(
            inputs=[self.x_inpt_single,self.x_inpt,y,Reward,lr],
            outputs=[cost],
            on_unused_input='warn',
            updates=updates)

        self.classification_results = sigmoid(self.output_layer_all)
        #self.classification_results = self.policy

        pre_lr = T.fscalar()
        lable = T.ivector()

        pre_cost = (- T.sum(T.log(self.classification_results + 1e-6 )*lable)\
                    - T.sum(T.log(1-self.classification_results+ 1e-6 )*(1-lable)))/(T.sum(lable) + T.sum(1-lable))\
                    + lmbda_l2*l2_norm_squared
        #pre_cost = - T.sum(T.log(self.classification_results + 1e-7 )*lable)/(T.sum(lable)+1)\
        #            - T.sum(T.log(1-self.classification_results+ 1e-7 )*(1-lable))/(T.sum(1-lable)+1)\
        #            + lmbda_l2*l2_norm_squared

        pregrads = T.grad(pre_cost, self.params)
        #max_norm = 5.0
        #pregrads = [lasagne.updates.norm_constraint(grad, max_norm, range(grad.ndim)) for grad in pregrads]
        clip_grad = 5.0
        pre_cgrads = [T.clip(g,-clip_grad, clip_grad) for g in pregrads]

        #pre_updates = lasagne.updates.rmsprop(pregrads, self.params, learning_rate=0.0001)
        pre_updates = lasagne.updates.rmsprop(pre_cgrads, self.params, learning_rate=pre_lr)

        self.pre_train_step = theano.function(
            inputs=[self.x_inpt_single,self.x_inpt,lable,pre_lr],
            outputs=[pre_cost],
            on_unused_input='warn',
            updates=pre_updates)

        self.pre_predict = theano.function(
            inputs=[self.x_inpt_single,self.x_inpt,lable],
            outputs=[self.classification_results],
            on_unused_input='warn')

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

    lable = [0,1,0]
    pre_lr = 5
    print r.pre_predict(x_sinlge,zp_x,lable)
    #r.show_para()
    print r.pre_train_step(x_sinlge,zp_x,lable,pre_lr)
    print r.pre_train_step(x_sinlge,zp_x,lable,pre_lr)
    print r.pre_train_step(x_sinlge,zp_x,lable,pre_lr)
    print r.pre_predict(x_sinlge,zp_x,lable)
    #r.show_para()

if __name__ == "__main__":
    main()
