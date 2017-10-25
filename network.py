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
        ## embedding = 832 for cn, 650 for en

        dropout_prob = T.scalar("probability of dropout")

        self.params = []

        self.x_inpt = T.matrix("input_pair_embeddings")

        w_h_1,b_h_1 = init_weight(n_inpt,n_hidden,pre="inpt_layer",ones=False) 
        self.params += [w_h_1,b_h_1]

        self.hidden_layer = tanh(T.dot(self.x_inpt,w_h_1) + b_h_1)

        w_h_2,b_h_2 = init_weight(n_hidden,1,pre="output_layer",ones=False) 
        self.params += [w_h_2,b_h_2]

        self.output_layer = tanh(T.dot(self.hidden_layer,w_h_2) + b_h_2)

        self.policy = softmax(self.output_layer.flatten())[0]

        self.get_out = theano.function(
            inputs=[self.x_inpt],
            outputs=[self.policy],
            on_unused_input='warn')

        lr = T.scalar()
        Reward = T.scalar("Reward")
        y = T.iscalar('classification')

        cost = (-Reward) * self.policy[y]

        self.get_cost = theano.function(
            inputs=[self.x_inpt,y,Reward],
            outputs=[cost],
            on_unused_input='warn')

        updates = lasagne.updates.sgd(cost, self.params, lr)

        self.train_step = theano.function(
            inputs=[self.x_inpt,y,Reward,lr],
            outputs=[cost],
            on_unused_input='warn',
            updates=updates)

    def show_para(self):
        for para in self.params:
            print >> sys.stderr, para,para.get_value() 

def main():
    r = NetWork(3,2)

    zp_x = [[2,3,4],[2,98,4]]
    print r.get_out(zp_x)[0][0]
    print r.get_out(zp_x)[0][1]
    #print r.get_cost(zp_x,0,100)
    #print r.get_cost(zp_x,1)
    r.train_step(zp_x,0,0.2,5)
    r.train_step(zp_x,0,0.2,5)
    r.train_step(zp_x,0,0.2,5)
    r.train_step(zp_x,1,1,5)
    print r.get_out(zp_x)[0]

if __name__ == "__main__":
    main()
