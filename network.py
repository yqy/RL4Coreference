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
    def __init__(self,n_inpt,n_hidden):
        ## input = 1738 for cn 1374 for en
        ## embedding for each mention = 855 for cn, 673 for en
        ## pair_feature = 28

        activate=ReLU

        dropout_prob = T.scalar("probability of dropout")

        self.params = []

        self.x_inpt = T.matrix("input_pair_embeddings")

        w_h_1,b_h_1 = init_weight(n_inpt,n_hidden,pre="inpt_layer",special=True,ones=False) 
        self.params += [w_h_1,b_h_1]

        self.hidden_layer_1 = activate(T.dot(self.x_inpt,w_h_1) + b_h_1)


        w_h_2,b_h_2 = init_weight(n_hidden,n_hidden/2,pre="hidden_layer_1",special=True,ones=False) 
        self.params += [w_h_2,b_h_2]

        self.hidden_layer_2_ = activate(T.dot(self.hidden_layer_1,w_h_2) + b_h_2)

        w_h_2_,b_h_2_ = init_weight(n_hidden/2,n_hidden/2,pre="hidden_layer_2",special=True,ones=False) 
        self.params += [w_h_2_,b_h_2_]

        self.hidden_layer_2 = activate(T.dot(self.hidden_layer_2_,w_h_2_) + b_h_2_)

        w_h_3,b_h_3 = init_weight(n_hidden/2,1,pre="output_layer",special=True,ones=False) 
        self.params += [w_h_3,b_h_3]

        self.output_layer = activate(T.dot(self.hidden_layer_2,w_h_3) + b_h_3)

        self.policy = softmax(self.output_layer.flatten())[0]

        self.predict = theano.function(
            inputs=[self.x_inpt],
            outputs=[self.policy],
            #allow_input_downcast=True,
            on_unused_input='warn')

        lr = T.scalar()
        Reward = T.scalar("Reward")
        y = T.iscalar('classification')

        l2_norm_squared = sum([(abs(w)).sum() for w in self.params])
        lmbda_l2 = 0.000001

        self.get_weight_sum = theano.function(inputs=[],outputs=[l2_norm_squared])

        cost = (-Reward) * T.log(self.policy[y] + 1e-7)\
                + lmbda_l2*l2_norm_squared

        grads = T.grad(cost, self.params)
        #grads = [lasagne.updates.norm_constraint(grad, max_norm, range(grad.ndim)) for grad in grads]
        #updates = lasagne.updates.rmsprop(grads, self.params, learning_rate=0.0001)
        updates = lasagne.updates.rmsprop(grads, self.params, learning_rate=lr)

        self.train_step = theano.function(
            inputs=[self.x_inpt,y,Reward,lr],
            outputs=[cost],
            on_unused_input='warn',
            updates=updates)

        self.classification_results = sigmoid(self.output_layer.flatten())
        pre_lr = T.scalar()
        lable = T.vector()

        pre_cost = - T.sum(T.log(self.classification_results + 1e-7 )*lable)/(T.sum(lable)+1)\
                    - T.sum(T.log(1-self.classification_results+ 1e-7 )*(1-lable))/(T.sum(1-lable)+1)\
                    + lmbda_l2*l2_norm_squared

        pregrads = T.grad(pre_cost, self.params)
        #pregrads = [lasagne.updates.norm_constraint(grad, max_norm, range(grad.ndim)) for grad in pregrads]
        #pre_updates = lasagne.updates.rmsprop(pregrads, self.params, learning_rate=0.0001)
        pre_updates = lasagne.updates.rmsprop(pregrads, self.params, learning_rate=pre_lr)

        self.pre_train_step = theano.function(
            inputs=[self.x_inpt,lable,pre_lr],
            outputs=[pre_cost],
            on_unused_input='warn',
            updates=pre_updates)

        self.pre_predict = theano.function(
            inputs=[self.x_inpt,lable],
            outputs=[self.classification_results],
            on_unused_input='warn')


    def show_para(self):
        for para in self.params:
            print >> sys.stderr, para,para.get_value() 

def main():
    r = NetWork(3,2)

    zp_x = [[2,3,4],[2,98,4]]
    print list(r.predict(zp_x)[0])
    #print r.predict(zp_x)[0][0]
    #print r.predict(zp_x)[0][1]
    #r.train_step(zp_x,0,0.2,5)
    #r.train_step(zp_x,0,0.2,5)
    #r.train_step(zp_x,0,0.2,5)
    #r.train_step(zp_x,1,1,5)
    #print r.predict(zp_x)[0]
    lable = [0,1]
    pre_lr = 0.1
    print r.pre_predict(zp_x,lable)
    print r.pre_train_step(zp_x,lable,pre_lr)
    print r.pre_train_step(zp_x,lable,pre_lr)
    print r.pre_train_step(zp_x,lable,pre_lr)
    print r.pre_predict(zp_x,lable)

if __name__ == "__main__":
    main()
