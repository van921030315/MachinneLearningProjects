########################################################################
#######       DO NOT MODIFY, DEFINITELY READ ALL OF THIS         #######
########################################################################

import numpy as np
import cnn_arc2
import pickle
import copy
import random
import preprocessing
def predict(params, layers, xtest, ytest):
  # load parameter from lenet.mat
  pickle_path = 'lenet.mat'
  pickle_file = open(pickle_path, 'rb')
  params = pickle.load(pickle_file)
  labels, msg = cnn_arc2.conv_net(params, layers, xtest, ytest, predict=True)
  if msg != "Predict Result":
    print "Incorrect output format"
  return labels


def get_lenet(batch_size, predict = False):
  """Define LeNet

  Explanation of parameters:
  type: layer type, supports convolution, pooling, relu
  channel: input channel
  num: output channel
  k: convolution kernel width (== height)
  group: split input channel into several groups, not used in this assignment
  """

  layers = {}
  layers[1] = {}
  layers[1]['type'] = 'DATA'
  layers[1]['height'] = 7
  layers[1]['width'] = 90
  layers[1]['channel'] = 1
  layers[1]['batch_size'] = 32
  if predict:
    layers[1]['batch_size'] = batch_size

  layers[2] = {}
  layers[2]['type'] = 'CONV'
  layers[2][1] = {}
  layers[2][1]['num'] = 30
  layers[2][1]['kh'] = 6
  layers[2][1]['kw'] = 90
  layers[2][1]['strideh'] = 1
  layers[2][1]['stridew'] = 1
  layers[2][1]['pad'] = 0
  layers[2][1]['group'] = 1

  layers[2][2] = {}
  layers[2][2]['num'] = 30
  layers[2][2]['kh'] = 5
  layers[2][2]['kw'] = 90
  layers[2][2]['strideh'] = 1
  layers[2][2]['stridew'] = 1
  layers[2][2]['pad'] = 0
  layers[2][2]['group'] = 1

  layers[2][3] = {}
  layers[2][3]['num'] = 30
  layers[2][3]['kh'] = 4
  layers[2][3]['kw'] = 90
  layers[2][3]['strideh'] = 1
  layers[2][3]['stridew'] = 1
  layers[2][3]['pad'] = 0
  layers[2][3]['group'] = 1

  layers[2][4] = {}
  layers[2][4]['num'] = 30
  layers[2][4]['kh'] = 3
  layers[2][4]['kw'] = 90
  layers[2][4]['strideh'] = 1
  layers[2][4]['stridew'] = 1
  layers[2][4]['pad'] = 0
  layers[2][4]['group'] = 1

  layers[2][5] = {}
  layers[2][5]['num'] = 30
  layers[2][5]['kh'] = 2
  layers[2][5]['kw'] = 90
  layers[2][5]['strideh'] = 1
  layers[2][5]['stridew'] = 1
  layers[2][5]['pad'] = 0
  layers[2][5]['group'] = 1
  '''
  layers[2][4] = {}
  layers[2][4]['num'] = 30
  layers[2][4]['kh'] = 3
  layers[2][4]['kw'] = 90
  layers[2][4]['strideh'] = 1
  layers[2][4]['stridew'] = 1
  layers[2][4]['pad'] = 0
  layers[2][4]['group'] = 1

  layers[2][5] = {}
  layers[2][5]['num'] = 30
  layers[2][5]['kh'] = 2
  layers[2][5]['kw'] = 90
  layers[2][5]['strideh'] = 1
  layers[2][5]['stridew'] = 1
  layers[2][5]['pad'] = 0
  layers[2][5]['group'] = 1

  layers[2][6] = {}
  layers[2][6]['num'] = 30
  layers[2][6]['kh'] = 2
  layers[2][6]['kw'] = 90
  layers[2][6]['strideh'] = 1
  layers[2][6]['stridew'] = 1
  layers[2][6]['pad'] = 0
  layers[2][6]['group'] = 1
  '''

  layers[3] = {}
  layers[3]['type'] = 'POOLING'
  layers[3][1] = {}
  layers[3][1]['kh'] = 2
  layers[3][1]['kw'] = 1
  layers[3][1]['strideh'] = 1
  layers[3][1]['stridew'] = 1
  layers[3][1]['pad'] = 0

  layers[3][2] = {}
  layers[3][2]['kh'] = 3
  layers[3][2]['kw'] = 1
  layers[3][2]['strideh'] = 1
  layers[3][2]['stridew'] = 1
  layers[3][2]['pad'] = 0

  layers[3][3] = {}
  layers[3][3]['kh'] = 4
  layers[3][3]['kw'] = 1
  layers[3][3]['strideh'] = 1
  layers[3][3]['stridew'] = 1
  layers[3][3]['pad'] = 0
  
  layers[3][4] = {}
  layers[3][4]['kh'] = 5
  layers[3][4]['kw'] = 1
  layers[3][4]['strideh'] = 1
  layers[3][4]['stridew'] = 1
  layers[3][4]['pad'] = 0

  layers[3][5] = {}
  layers[3][5]['kh'] = 6
  layers[3][5]['kw'] = 1
  layers[3][5]['strideh'] = 1
  layers[3][5]['stridew'] = 1
  layers[3][5]['pad'] = 0
  '''
  layers[3][6] = {}
  layers[3][6]['kh'] = 4
  layers[3][6]['kw'] = 1
  layers[3][6]['strideh'] = 1
  layers[3][6]['stridew'] = 1
  layers[3][6]['pad'] = 0
  '''
  layers[4] = {}
  layers[4]['type'] = 'SUM'
  layers[4]['num'] = 150
  layers[4]['init_type'] = 'uniform'

  layers[5] = {}
  layers[5]['type'] = 'IP'
  layers[5]['num'] = 38
  layers[5]['init_type'] = 'uniform'

  layers[6] = {}
  layers[6]['type'] = 'RELU'

  layers[7] = {}
  layers[7]['type'] = 'LOSS'
  layers[7]['num'] = 38
  return layers


def main():
  # define lenet
  layers = get_lenet(157850, True)

  # load data
  # change the following value to true to load the entire dataset
  fullset = False
  predict_data = True
  xtrain, ytrain, xtest, ytest = preprocessing.load_data(predict_data, iobtag = False)

  # xtrain = np.hstack([xtrain, xval])
  # ytrain = np.hstack([ytrain, yval])
  m_train = xtrain.shape[1]
  print "finish loading data"
  # cnn parameters
  batch_size = 32
  if predict_data:
    batch_size = m_train
  mu = 0.9
  epsilon = 0.01
  gamma = 0.0001
  power = 0.75
  weight_decay = 0.0005
  w_lr = 1
  b_lr = 2

  test_interval = 500
  display_interval = 100
  snapshot = 2000
  max_iter = 10000


  # initialize parameters
  params = cnn_arc2.init_convnet(layers)
  param_winc = copy.deepcopy(params)
  param_pre = copy.deepcopy(params)
  if predict_data:
    pickle_file = open('output.mat', 'wb')
    output = predict(params, layers, xtrain, ytrain)
    pickle.dump(output, pickle_file)
    return 0

  for l_idx in range(1, len(layers)):
    if l_idx == 1 or l_idx == 2:
      for j in range(1, len(layers[l_idx + 1])):
        #print l_idx, j, len(layers[l_idx + 1])
        param_winc[l_idx][j]['w'] = np.zeros(param_winc[l_idx][j]['w'].shape)
        param_winc[l_idx][j]['b'] = np.zeros(param_winc[l_idx][j]['b'].shape)
      continue
    param_winc[l_idx]['w'] = np.zeros(param_winc[l_idx]['w'].shape)
    param_winc[l_idx]['b'] = np.zeros(param_winc[l_idx]['b'].shape)

  # learning iterations
  indices = range(m_train)
  random.shuffle(indices)
  for step in range(max_iter):
    # get mini-batch and setup the cnn with the mini-batch
    start_idx = step * batch_size % m_train
    end_idx = (step+1) * batch_size % m_train
    if start_idx > end_idx:
      random.shuffle(indices)
      continue
    idx = indices[start_idx: end_idx]

    [cp, param_grad] = cnn_arc2.conv_net(params,
                                          layers,
                                          xtrain[:, idx],
                                          ytrain[idx])
    break
    # we have different epsilons for w and b
    w_rate = cnn_arc2.get_lr(step, epsilon*w_lr, gamma, power)
    b_rate = cnn_arc2.get_lr(step, epsilon*b_lr, gamma, power)
    params, param_winc = cnn_arc2.sgd_momentum(w_rate,
                           b_rate,
                           mu,
                           weight_decay,
                           params,
                           param_winc,
                           param_grad)

    # display training loss
    if (step+1) % display_interval == 0:
      print 'cost = %f training_percent = %f' % (cp['cost'], cp['percent'])

    # display test accuracy
    if (step+1) % test_interval == 0:
      layers[1]['batch_size'] = xtest.shape[1]
      cptest, _ = cnn_arc2.conv_net(params, layers, xtest, ytest)
      layers[1]['batch_size'] = 32
      print '\ntest accuracy: %f\n' % (cptest['percent'])

    # save params peridocally to recover from any crashes
    if (step+1) % snapshot == 0:
      pickle_path = 'lenet.mat'
      pickle_file = open(pickle_path, 'wb')
      pickle.dump(params, pickle_file)
      pickle_file.close()


if __name__ == '__main__':
  main()
