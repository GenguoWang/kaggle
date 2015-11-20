require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers

model = nn.Sequential()
model:add(nn.SpatialConvolution(1,32,5,5))
model:add(nn.Tanh())
model:add(nn.SpatialMaxPooling(3,3,3,3))
model:add(nn.SpatialConvolution(32,64,5,5))
model:add(nn.Tanh())
model:add(nn.SpatialMaxPooling(2,2,2,2))

ninputs = 64*2*2
nhiddens = 200
model:add(nn.Reshape(ninputs))
model:add(nn.Linear(ninputs,nhiddens))
model:add(nn.Tanh())
model:add(nn.Linear(nhiddens,noutputs))

print(model)
