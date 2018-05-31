import numpy as np
'''
# data I/O
data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
'''
# hyperparameters
hidden_size = 8 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

vocab_size = 11
# model parameters


#inputs
f1 = open('inputs.txt','r')
f2 = open('targets.txt','r')

out_file = open('output.txt','w')

all_inputs = f1.readlines()
all_targets = f2.readlines()

train_input = all_inputs[:len(all_inputs)*80/100]
test_input = all_inputs[len(all_inputs)*80/100:]

train_target = all_targets[:len(all_inputs)*80/100]
test_target = all_targets[len(all_inputs)*80/100:]


#forward RNN Class
Wxh_f = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh_f = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why_f = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh_f = np.zeros((hidden_size, 1)) # hidden bias
#by = np.zeros((vocab_size, 1)) # output bias

#backward RNN Class
Wxh_b = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh_b = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why_b = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh_b = np.zeros((hidden_size, 1)) # hidden bias
#by_b = np.zeros((vocab_size, 1)) # output bias

#output layer bias is one vector
by = np.zeros((vocab_size, 1)) # output bias
pass_cases = 0
def lossFun(inputs, targets, hprev_f,hprev_b, op_type):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  global pass_cases
  xs_f, hs_f, xs_b,hs_b,ys, ps = {}, {}, {}, {},{}, {}
  hs_f[-1] = np.copy(hprev_f)
  hs_b[-1] = np.copy(hprev_b)
  loss = 0
  target=[]
  # forward pass
  for t in xrange(len(inputs)):
    xs_f[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs_f[t][inputs[t]] = 1


    #forward RNN
    hs_f[t] = np.tanh(np.dot(Wxh_f, xs_f[t]) + np.dot(Whh_f, hs_f[t-1]) + bh_f) # hidden state

    #loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
    #print "activations forward", hs_f[t]

    #backward RNN
  rev = inputs[::-1]
  for t in (xrange(len(rev))):
    xs_b[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs_b[t][inputs[t]] = 1
  #print "input",xs_b

  #backward RNN
  for t in (xrange(len(rev))):
    #nt = len(inputs)-t-1
    #print "test",hs_b[t-1]
    hs_b[t] = np.tanh(np.dot(Wxh_b, xs_b[t]) + np.dot(Whh_b, hs_b[t-1]) + bh_b) # hidden state
    #ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    #ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    #loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
    #print "activations backward", hs_b[t]

  #output layer
  send_target =[]
  for t in (xrange(len(inputs))):
    ys[t] = np.tanh(np.dot(Why_f, hs_f[t]) + np.dot(Why_b, hs_b[t]) + by) # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    send_target.append(ps[t].argmax())
    #print "ps",ps[t][targets[t],0]
    #print max(ps[t]),"t",t
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
    #loss += 1-(ps[target.index(1)])
  #print "probability",ps,len(ps),len(ps[0])

  # backward pass: compute gradients going backwards
  #for output layer

  #print "loss",loss
  #if n % 100 == 0:
    #print "target : ",targets, " ... predicted : ", send_target
  #exit(0)
  # ##check if from test data and return with result if yes
  if op_type=="predict":
    send_target =[]
    for t in (xrange(len(inputs))):
      #ys[t] = np.tanh(np.dot(Why_f, hs_f[t]) + np.dot(Why_b, hs_b[t]) + by) # unnormalized log probabilities for next chars
      #ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
      send_target.append(ps[t].argmax())

    #compare target with ps

    correct_count = 0
    for i in range(len(send_target)):
      if send_target[i]==targets[i]:
        correct_count+=1
    accuracy = correct_count/float(len(send_target))
    print "target: " ,targets, " ... predicted : ", send_target, "   local acc : ", accuracy

    if correct_count==len(send_target):
      pass_cases+=1

    return


  # ##else

  #fRNN
  dWxh_f, dWhh_f, dWhy_f = np.zeros_like(Wxh_f), np.zeros_like(Whh_f), np.zeros_like(Why_f)
  dbh_f, dby = np.zeros_like(bh_f), np.zeros_like(by)
  dhnext_f = np.zeros_like(hs_f[0])
  #bRNN
  dWxh_b, dWhh_b, dWhy_b = np.zeros_like(Wxh_b), np.zeros_like(Whh_b), np.zeros_like(Why_b)
  dbh_b, dby = np.zeros_like(bh_b), np.zeros_like(by)
  dhnext_b = np.zeros_like(hs_b[0])

  for t in reversed(xrange(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy_f += np.dot(dy, hs_f[t].T)
    dWhy_b += np.dot(dy, hs_b[t].T)
    dby += dy
    dh_f = np.dot(Why_f.T, dy) + dhnext_f # backprop into h
    dh_b = np.dot(Why_b.T, dy) + dhnext_b # backprop into h
    dhraw_f = (1 - hs_f[t] * hs_f[t]) * dh_f # backprop through tanh nonlinearity
    dhraw_b = (1 - hs_b[t] * hs_b[t]) * dh_b
    dbh_f += dhraw_f
    dbh_b += dhraw_b
    dWxh_f += np.dot(dhraw_f, xs_f[t].T)
    dWxh_b += np.dot(dhraw_b, xs_b[t].T)
    dWhh_f += np.dot(dhraw_f, hs_f[t-1].T)
    dWhh_b += np.dot(dhraw_b, hs_b[t-1].T)
    dhnext_f = np.dot(Whh_f.T, dhraw_f)
    dhnext_b = np.dot(Whh_b.T, dhraw_b)

  for dparam in [dWxh_f, dWhh_f, dWhy_f, dbh_f, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients

  for dparam in [dWxh_b, dWhh_b, dWhy_b, dbh_b, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients


  #print "fforward",loss, hs_f[len(inputs)-1]
  #print "backward",loss, hs_b[len(inputs)-1]
  #exit(0)
  #return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]
  return loss, dWxh_f,dWxh_b, dWhh_f, dWhh_b, dWhy_f, dWhy_b, dbh_f, dbh_b, dby, hs_f[len(inputs)-1], hs_b[len(inputs)-1], ps, send_target

#predict
#pass_cases = 0
def predict(inputs, targets, hprev_f,hprev_b):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs_f, hs_f, xs_b,hs_b,ys, ps = {}, {}, {}, {},{}, {}
  hs_f[-1] = np.copy(hprev_f)
  hs_b[-1] = np.copy(hprev_b)
  target=[]
  # forward pass
  for t in xrange(len(inputs)):
    xs_f[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs_f[t][inputs[t]] = 1


    #forward RNN
    hs_f[t] = np.tanh(np.dot(Wxh_f, xs_f[t]) + np.dot(Whh_f, hs_f[t-1]) + bh_f) # hidden state

    #loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
    #print "activations forward", hs_f[t]

    #backward RNN
  rev = inputs[::-1]
  for t in (xrange(len(rev))):
    xs_b[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs_b[t][inputs[t]] = 1
  #print "input",xs_b

  #backward RNN
  for t in (xrange(len(rev))):
    #nt = len(inputs)-t-1
    #print "test",hs_b[t-1]
    hs_b[t] = np.tanh(np.dot(Wxh_b, xs_b[t]) + np.dot(Whh_b, hs_b[t-1]) + bh_b) # hidden state
    #ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    #ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    #loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
    #print "activations backward", hs_b[t]

  #output layer
  send_target =[]
  for t in (xrange(len(inputs))):
    ys[t] = np.tanh(np.dot(Why_f, hs_f[t]) + np.dot(Why_b, hs_b[t]) + by) # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    send_target.append(ps[t].argmax())

  #compare target with ps

  correct_count = 0
  for i in range(len(send_target)):
    if send_target[i]==targets[i]:
      correct_count+=1
  accuracy = correct_count/len(send_target)
  print "target: " ,targets, " ... predicted : ", send_target, "   local acc : ", accuracy

  if correct_count==len(send_target):
    pass_cases+=1

  #return ps


def sample(h, seed_ix, n):
  """
  sample a sequence of integers from the model
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

n, p = 0, 0
#fRNN
mWxh_f, mWhh_f, mWhy_f = np.zeros_like(Wxh_f), np.zeros_like(Whh_f), np.zeros_like(Why_f)
mbh_f, mby_f = np.zeros_like(bh_f), np.zeros_like(by) # memory variables for Adagrad
#bRNN
mWxh_b, mWhh_b, mWhy_b = np.zeros_like(Wxh_b), np.zeros_like(Whh_b), np.zeros_like(Why_b)
mbh_b, mby_b = np.zeros_like(bh_b), np.zeros_like(by) # memory variables for Adagrad

smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

out_hprev_f =0
out_hprev_b =0
out_loss = 0
while n<len(train_input):
  global out_hprev_f
  global out_hprev_b
  global out_loss
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  '''
  if p+seq_length+1 >= len(data) or n == 0: '''
  hprev_f = np.zeros((hidden_size,1)) # reset fRNN memory
  hprev_b = np.zeros((hidden_size,1)) # reset bRNN memory
  p = 0 # go from start of data
  #inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  #targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # we should make it one-hot vector

  inputs = [int(i) for i in train_input[n].strip().split(' ')]
  targets = [int(i) for i in train_target[n].strip().split(' ')]
  # ##print inputs, targets
  #out_file.write(inputs, targets)
  # sample from the model now and then
  '''
  if n % 100 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print '----\n %s \n----' % (txt, )'''

  # forward seq_length characters through the net and fetch gradient
  epoch =0
  while epoch<10:
	  #fRNN
    loss, dWxh_f,dWxh_b, dWhh_f, dWhh_b, dWhy_f, dWhy_b, dbh_f, dbh_b, dby, hprev_f, hprev_b, ps, got_target = lossFun(inputs, targets, hprev_f,hprev_b,"loss")
    #print "epoch", epoch, "predicted",ps
	  #bRNN
	  #loss, dWxh_b, dWhh_b, dWhy_b, dbh_b, dby_b, hprev_b = lossFun(inputs, targets, hprev_b)

	  # perform parameter update with Adagrad
    for fparam, bparam, fdparam, bdparam, fmem, bmem in zip([Wxh_f, Whh_f, Why_f, bh_f, by],
	  								[Wxh_b, Whh_b, Why_b, bh_b, by],
		                            [dWxh_f, dWhh_f, dWhy_f, dbh_f, dby],
		                            [dWxh_b, dWhh_b, dWhy_b, dbh_b, dby],
		                            [mWxh_f, mWhh_f, mWhy_f, mbh_f, mby_f],
		                            [mWxh_b, mWhh_b, mWhy_b, mbh_b, mby_b]):
		fmem += fdparam * fdparam
		bmem += bdparam * bdparam
		fparam += -learning_rate * fdparam / np.sqrt(fmem + 1e-8) # adagrad update
		bparam += -learning_rate * bdparam / np.sqrt(bmem + 1e-8) # adagrad update

    epoch+=1

    out_hprev_f = hprev_f
    out_hprev_b = hprev_b
    out_loss = loss

  smooth_loss = smooth_loss * 0.999 + out_loss * 0.001
  if n % 100 == 0:
    print 'iter %d, loss: %f ' % (n, smooth_loss) # print progress
    print 'target : ',targets, " ... predict : ", got_target


  p += seq_length # move data pointer
  n += 1 # iteration counter


print " total loss after training : ", out_loss
for i in range(len(test_input)):
  inputs = [int(j) for j in test_input[i].strip().split(' ')]
  targets = [int(j) for j in test_target[i].strip().split(' ')]
  lossFun(inputs, targets,out_hprev_f,out_hprev_b, "predict")

print "no of pass cases : ", pass_cases," ... % accuracy : ", pass_cases/float(len(test_input))*100