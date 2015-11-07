require 'torch'
require 'nn'

model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

print(criterion)
