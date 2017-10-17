#coding=utf8
from NetworkComponet import *

#theano.config.exception_verbosity="high"
#theano.config.optimizer="fast_compile"

'''
Deep neural network for AZP resolution
a kind of Memory Network
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
    def __init__(self,n_hidden,embedding_dimention=50,feature_dimention=61):

        ##n_in: sequence lstm 的输入维度
        ##n_hidden: lstm for candi and zp 的隐层维度

        dropout_prob = T.scalar("probability of dropout")

        self.params = []

        self.zp_x_pre = T.tensor3("zp_x_pre")
        self.zp_mask_pre = T.matrix("zp_mask_pre")
        self.zp_x_post = T.tensor3("zp_x_post")
        self.zp_mask_post = T.matrix("zp_mask_post")

        zp_x_pre_dropout = dropout_from_layer(self.zp_x_pre,dropout_prob)
        zp_nn_pre = LSTM_batch(embedding_dimention,n_hidden,zp_x_pre_dropout,self.zp_mask_pre,prefix="zp_pre")
        self.params += zp_nn_pre.params

        zp_x_post_dropout = dropout_from_layer(self.zp_x_post,dropout_prob)
        zp_nn_post = LSTM_batch(embedding_dimention,n_hidden,zp_x_post_dropout,self.zp_mask_post,prefix="zp_post")
        self.params += zp_nn_post.params

        self.zp_nn_output = dropout_from_layer(T.concatenate((zp_nn_pre.nn_out,zp_nn_post.nn_out),axis=1),dropout_prob)

        ### zp : batch*2d

        ### get sequence output for NP stacks ###

        ### NP -- forward direction 
        self.np_x_stack = T.tensor3("np_x_stack")
        np_x_stack_dropout = dropout_from_layer(self.np_x_stack,dropout_prob)
        self.np_mask_stack = T.matrix("np_mask_stack")

        ### NP -- LSTM modeling all the NPs
        np_nn_stack = LSTM_batch(embedding_dimention*2,n_hidden*2,np_x_stack_dropout,self.np_mask_stack,prefix="np_stack")
        self.params += np_nn_stack.params
        self.np_nn_stack_output = dropout_from_layer(np_nn_stack.nn_out , dropout_prob)

        ### np: batch * 2d

        ### current NP
        #forward
        self.np_current_x_pre = T.tensor3("np_current_x_pre")
        np_current_x_pre_dropout = dropout_from_layer(self.np_current_x_pre,dropout_prob)
        self.np_mask_x_pre = T.matrix("mask")

        self.np_current_c_pre = T.tensor3("np_current_c_pre")
        np_current_c_pre_dropout = dropout_from_layer(self.np_current_c_pre,dropout_prob)
        self.np_mask_c_pre = T.matrix("mask")

        np_nn_current_pre = sub_LSTM_batch(embedding_dimention,n_hidden,np_current_x_pre_dropout,np_current_c_pre_dropout,self.np_mask_x_pre,self.np_mask_c_pre,prefix="np_current_pre")
        self.params += np_nn_current_pre.params
        self.np_nn_current_pre_output = np_nn_current_pre.nn_out

        #backward
        self.np_current_x_post = T.tensor3("np_current_x_post")
        np_current_x_post_dropout = dropout_from_layer(self.np_current_x_post,dropout_prob)
        self.np_mask_x_post = T.matrix("mask")

        self.np_current_c_post = T.tensor3("np_current_c_post")
        np_current_c_post_dropout = dropout_from_layer(self.np_current_c_post,dropout_prob)
        self.np_mask_c_post = T.matrix("mask")

        np_nn_current_post = sub_LSTM_batch(embedding_dimention,n_hidden,np_current_x_post_dropout,np_current_c_post_dropout,self.np_mask_x_post,self.np_mask_c_post,prefix="np_current_post")
        self.params += np_nn_current_post.params
        self.np_nn_current_post_output = np_nn_current_post.nn_out

        self.np_nn_current_output = dropout_from_layer(T.concatenate((self.np_nn_current_pre_output,self.np_nn_current_post_output),axis=1),dropout_prob)

        ### current np : batch * 2d

        #calculate attention

        w_attention_zp,b_attention = init_weight(n_hidden*2,2,pre="attention_zp",ones=False) 
        self.params += [w_attention_zp,b_attention]

        w_attention_current_np,b_u = init_weight(n_hidden*2,2,pre="attention_np",ones=False) 
        self.params += [w_attention_current_np]

        w_attention_np_all,b_u = init_weight(n_hidden*2,2,pre="attention_np_all",ones=False) 
        self.params += [w_attention_np_all]

        self.calcu_attention = tanh(T.dot(self.np_nn_stack_output,w_attention_np_all) + T.dot(self.zp_nn_output,w_attention_zp) + T.dot(self.np_nn_current_output,w_attention_current_np) + b_attention)

        self.attention = softmax(self.calcu_attention)

        self.out = self.attention


        lr = T.scalar()
        Reward = T.vector("Reward")
        y = T.ivector('classification')
        #t = T.bscalar()

        cost = (-Reward * T.log(self.out[T.arange(y.shape[0]), y])).mean()
        #cost = - Reward * T.log(self.out[t])

        self.get_out = theano.function(
            inputs=[self.zp_x_pre,self.zp_x_post,self.zp_mask_pre,self.zp_mask_post,
                    self.np_x_stack,self.np_mask_stack,
                    self.np_current_x_pre,self.np_current_c_pre,self.np_current_x_post,self.np_current_c_post,
                    self.np_mask_x_pre,self.np_mask_c_pre,self.np_mask_x_post,self.np_mask_c_post,
                    dropout_prob,Reward,y],
            #outputs=[self.np_nn_stack_output],
            outputs=[self.out],
            #outputs=[cost],
            on_unused_input='warn')

        updates = lasagne.updates.sgd(cost, self.params, lr)

        self.train_step = theano.function(
            inputs=[self.zp_x_pre,self.zp_x_post,self.zp_mask_pre,self.zp_mask_post,
                    self.np_x_stack,self.np_mask_stack,
                    self.np_current_x_pre,self.np_current_c_pre,self.np_current_x_post,self.np_current_c_post,
                    self.np_mask_x_pre,self.np_mask_c_pre,self.np_mask_x_post,self.np_mask_c_post,
                    dropout_prob,Reward,y,lr],
            outputs=[cost],
            on_unused_input='warn',
            updates=updates)

        '''
        self.get_out = theano.function(inputs=[self.zp_x_pre,self.zp_x_post,self.np_x_pre,self.np_x_prec,self.np_x_post,self.np_x_postc,self.mask_pre,self.mask_prec,self.mask_post,self.mask_postc,dropout_prob],outputs=[self.out],on_unused_input='warn')
        
        l1_norm_squared = sum([(w**2).sum() for w in self.params])
        l2_norm_squared = sum([(abs(w)).sum() for w in self.params])

        lmbda_l1 = 0.0
        #lmbda_l2 = 0.001
        lmbda_l2 = 0.0

        t = T.bvector()
        cost = -(T.log((self.out*t).sum()))

        lr = T.scalar()
        updates = lasagne.updates.sgd(cost, self.params, lr)
        #updates = lasagne.updates.rmsprop(cost, self.params,0.001)
        #updates = lasagne.updates.adadelta(cost, self.params,0.01)
        #updates = lasagne.updates.adam(cost, self.params,0.001)

        self.train_step = theano.function(
            #inputs=[self.zp_x_pre,self.zp_x_post,self.np_x_pre,self.np_x_prec,self.np_x_post,self.np_x_postc,self.mask_pre,self.mask_prec,self.mask_post,self.mask_postc,self.feature,t,lr,dropout_prob],
            inputs=[self.zp_x_pre,self.zp_x_post,self.np_x_pre,self.np_x_prec,self.np_x_post,self.np_x_postc,self.mask_pre,self.mask_prec,self.mask_post,self.mask_postc,t,lr,dropout_prob],
            outputs=[cost],
            on_unused_input='warn',
            updates=updates)
        '''
    def show_para(self):
        for para in self.params:
            print >> sys.stderr, para,para.get_value() 

def main():
            #inputs=[self.zp_x_pre,self.zp_x_post,zp_mask_pre,zp_mask_post
            #        self.np_x_stack,self.np_mask_stack,
            #        self.np_current_x_pre,self.np_current_c_pre,self.np_current_x_post,self.np_current_c_post,
            #        self.np_mask_x_pre,self.np_mask_c_pre,self.np_mask_x_post,self.np_mask_c_post,
            #        dropout_prob],

    r = NetWork(3,2)

    t = [0,1,0]
    zp_x = [[[2,3],[1,2],[2,3],[0,0]],[[1,5],[2,8],[7,6],[1,1]]]
    zp_mask = [[1,1,1,0],[1,1,1,1]]

    np_x = [[[1,2],[2,3],[3,1]],[[2,3],[1,2],[0,0]]]
    mask = [[1,1,1],[1,1,0]]

    npp_x = [[[1,2,2,3],[2,3,3,4]],[[3,3,3,3],[0,0,0,0]]]
    maskk = [[1,1],[1,0]]

    current_np_x = [[[1,2],[2,3],[4,4]],[[1,1],[2,2],[3,3]]]
    current_np_xc = [[[2,3],[4,4]],[[0,1],[1,0]]]
    current_mask_x = [[1,1,1],[1,1,1]]
    current_mask_c = [[1,1],[1,1]]

    y = [0,1]
    Reward = np.array([1,0.1323],dtype = np.float32)

    #print r.get_np_out(np_x,npp_x,np_x,npp_x,mask,maskk,mask,maskk)
    #print r.get_out(zp_x,zp_x,np_x,npp_x,np_x,npp_x,mask,maskk,mask,maskk)
    #print "Train"
    #r.train_step(zp_x,zp_x,np_x,npp_x,np_x,npp_x,mask,maskk,mask,maskk,t,5)
    #r.train_step(zp_x,zp_x,np_x,npp_x,np_x,npp_x,mask,maskk,mask,maskk,t,5)
    #r.train_step(zp_x,zp_x,np_x,npp_x,np_x,npp_x,mask,maskk,mask,maskk,t,5)

    lr = 0.5

    print r.get_out(zp_x,zp_x,zp_mask,zp_mask,npp_x,maskk,current_np_x,current_np_xc,current_np_x,current_np_xc,current_mask_x,current_mask_c,current_mask_x,current_mask_c,0.0,Reward,y)
    print r.train_step(zp_x,zp_x,zp_mask,zp_mask,npp_x,maskk,current_np_x,current_np_xc,current_np_x,current_np_xc,current_mask_x,current_mask_c,current_mask_x,current_mask_c,0.0,Reward,y,lr)
    print r.train_step(zp_x,zp_x,zp_mask,zp_mask,npp_x,maskk,current_np_x,current_np_xc,current_np_x,current_np_xc,current_mask_x,current_mask_c,current_mask_x,current_mask_c,0.0,Reward,y,lr)
    print r.train_step(zp_x,zp_x,zp_mask,zp_mask,npp_x,maskk,current_np_x,current_np_xc,current_np_x,current_np_xc,current_mask_x,current_mask_c,current_mask_x,current_mask_c,0.0,Reward,y,lr)
    print r.get_out(zp_x,zp_x,zp_mask,zp_mask,npp_x,maskk,current_np_x,current_np_xc,current_np_x,current_np_xc,current_mask_x,current_mask_c,current_mask_x,current_mask_c,0.0,Reward,y)

    #q = list(r.get_out(zp_x,zp_x,np_x,npp_x,np_x,npp_x,mask,maskk,mask,maskk)[0])
    #for num in q:
    #    print num

if __name__ == "__main__":
    main()
