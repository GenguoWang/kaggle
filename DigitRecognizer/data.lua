require 'nn'
require 'torch'
local obj = torch.load("data.t7")
dataTensor = obj.dataTensor
labelTensor = obj.labelTensor
testTensor = obj.testTensor

mean = dataTensor:mean()
stdv = dataTensor:std()
print(mean)
print(stdv)

dataTensor:add(-mean)
dataTensor:mul(1/stdv)

testTensor:add(-mean)
testTensor:mul(1/stdv)

labelTensor:add(1)
testSize = 400
trsize = dataTensor:size(1)-testSize
print("===train Data size:",trsize)
trainData = {
    data = dataTensor[{{1,trsize}}]:float(),
    labels = labelTensor[{{1,trsize}}],
    size = function() return trsize end
}
print(trainData)


tesize = testSize
testData = {
    data = dataTensor[{{1+trsize,trsize+tesize}}]:float(),
    labels = labelTensor[{{1+trsize,trsize+tesize}}],
    size = function() return tesize end
}
print(testData)

noutputs = 10
ninputs = 28*28
classes = {'0','1','2','3','4','5','6','7','8','9'}
