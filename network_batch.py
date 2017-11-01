#coding=utf8
from NetworkComponet import *
import numpy

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

        self.x_inpt = T.tensor3("input_pair_embeddings")
        self.x_inpt_single = T.matrix("input_pair_embeddings")
        self.x_mask = T.matrix("mask")

        w_h_1,b_h_1 = init_weight(n_inpt,n_hidden,pre="inpt_layer",ones=False) 
        self.params += [w_h_1,b_h_1]

        self.hidden_layer_1 = activate(T.dot(self.x_inpt,w_h_1) + b_h_1)

        w_h_2,b_h_2 = init_weight(n_hidden,n_hidden/2,pre="hidden_layer",ones=False) 
        self.params += [w_h_2,b_h_2]

        self.hidden_layer_2 = activate(T.dot(self.hidden_layer_1,w_h_2) + b_h_2)

        w_h_3,b_h_3 = init_weight(n_hidden/2,1,pre="output_layer",ones=False) 
        self.params += [w_h_3,b_h_3]

        self.output_layer = activate(T.dot(self.hidden_layer_2,w_h_3) + b_h_3)

        output_after_mask = T.switch(self.x_mask, self.output_layer.flatten(2), numpy.NINF)

        self.output_single = activate(T.dot(activate(T.dot(activate(T.dot(self.x_inpt_single,w_h_1) + b_h_1),w_h_2) + b_h_2),w_h_3) +b_h_3)
        self.policy_single = softmax(self.output_single.flatten())[0]
        self.predict = theano.function(
            inputs=[self.x_inpt_single],
            outputs=[self.policy_single],
            allow_input_downcast=True,
            on_unused_input='warn')

        self.policy = softmax(output_after_mask)

        self.predict_batch = theano.function(
            inputs=[self.x_inpt,self.x_mask],
            outputs=[self.policy],
            #outputs=[self.output_layer],
            allow_input_downcast=True,
            on_unused_input='warn')

        lr = T.scalar()
        Reward = T.vector("Reward")
        y = T.ivector('classification')

        cost = T.mean((-Reward) * T.log(self.policy[T.arange(y.shape[0]), y]))

        updates = lasagne.updates.sgd(cost, self.params, lr)

        self.train_step = theano.function(
            inputs=[self.x_inpt,self.x_mask,y,Reward,lr],
            outputs=[cost],
            on_unused_input='warn',
            allow_input_downcast=True,
            updates=updates)

    def show_para(self):
        for para in self.params:
            print >> sys.stderr, para,para.get_value() 

def main():
    r = NetWork(3,2)

    zp_x = [[[2,3,4],[2,98,4]]*2,[[2,98,4],[0,0,0]]*2]
    mask = [[1,1,1,1],[1,0,1,1]]
    print r.predict(zp_x,mask)[0]

    y = [3,2]
    re = [0.8,0.9]

    print r.predict_batch(zp_x,mask)[0][0]
    print r.predict_batch(zp_x,mask)[0][1]
    r.train_step(zp_x,mask,y,re,5)
    r.train_step(zp_x,mask,y,re,5)
    r.train_step(zp_x,mask,y,re,5)
    r.train_step(zp_x,mask,y,re,5)
    print r.predict_batch(zp_x,mask)

def test_switch():
    a = T.matrix()
    m = T.matrix()
    a_mask = T.switch(m, a, numpy.NINF)
    #a_mask = a
    b = softmax(a_mask)

    #f = theano.function(inputs=[a,m],outputs=[b])
    f = theano.function(inputs=[a,m],outputs=[b],on_unused_input='warn')
    x = [[2,2,3],[2,3,3]]
    m = [[1,1,0],[1,1,1]]
    print f(x,m)
    
if __name__ == "__main__":
    main()