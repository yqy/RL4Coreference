#python resolution.py -type nn_train -echos 20 -data /users1/qyyin/qyyin/ACL/data/train -embedding /users1/qyyin/cn/sogou.word2vec.100d.txt -test_data /users1/qyyin/qyyin/ACL/data/test > result
#python resolution.py -type nn_train_feature -echos 20 -data /users1/qyyin/qyyin/ACL/data/train -embedding /users1/qyyin/cn/sogou.word2vec.100d.txt -test_data /users1/qyyin/qyyin/ACL/data/test > result
#python resolution.py -type nn_train_feature -echos 20 -data /users1/qyyin/qyyin/ACL/data/train -embedding /users1/qyyin/cn/sogou.word2vec.100d.txt -test_data /users1/qyyin/qyyin/ACL/data/test -lr 0.003 > result
#python resolution.py -type nn_train -echos 50 -data /users1/qyyin/qyyin/ACL/data/test -embedding /users1/qyyin/cn/yqy.skip -test_data /users1/qyyin/qyyin/ACL/data/test -lr 0.003 > result
#python resolution.py -type nn_single -echos 20 -data /users1/qyyin/qyyin/ACL/data/train -embedding /users1/qyyin/cn/sogou.word2vec.50d.txt -test_data /users1/qyyin/qyyin/ACL/data/test -lr 0.01 -embedding_dimention 50 -batch 64 > result
#python resolution.py -type nn_update -echos 20 -data /users1/qyyin/qyyin/ACL/data/train -embedding /users1/qyyin/cn/embedding.ontoTrain.cbow -test_data /users1/qyyin/qyyin/ACL/data/test -lr 0.003 -embedding_dimention 50 > result
#THEANO_FLAGS='device=cpu' python resolution.py -type nn_update -echos 20 -data /users1/qyyin/qyyin/ACL/data/train -embedding /users1/qyyin/cn/embedding.sogouTrain.50 -test_data /users1/qyyin/qyyin/ACL/data/test -lr 0.003 -embedding_dimention 50 > result

python resolution.py -type nn_train -echos 5 -data ~/data/test/mz/ -embedding /Users/yqy/work/data/word2vec/embedding.ontonotes -test_data ~/data/test/mz/ -lr 0.003 -embedding_dimention 100

