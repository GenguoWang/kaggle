require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers

nhiddens = ninputs / 2

trainData.data = trainData.data:squeeze()
testData.data = testData.data:squeeze()
model = nn.Sequential()
model:add(nn.Reshape(ninputs),false)
model:add(nn.Linear(ninputs,nhiddens))
model:add(nn.Tanh())
model:add(nn.Linear(nhiddens,noutputs))

print(model)
