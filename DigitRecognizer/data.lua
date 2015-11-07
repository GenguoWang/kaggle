require 'nn'
require 'torch'
local csv = require 'csv'
local testFile = csv.open("test.csv")
local dataTb = {}
local testSize = 200
local trainSize = 4000
local i=0
for fields in testFile:lines() do
    if i> 0 then
  table.insert(dataTb, fields)
    end
    i=i+1
    if i > testSize then break end
end
testData = torch.Tensor(#dataTb,1,28,28)
for i=1,#dataTb do
    for j=1,28 do
        for k=1,28 do
            testData[{i,1,j,k}] = tonumber(dataTb[i][(j-1)*28+k])
        end
    end
end
local trainFile = csv.open("train.csv")
dataTb = {}
i=0
for fields in trainFile:lines() do
    if i> 0 then
          table.insert(dataTb, fields)
    end
    i=i+1
    if i > trainSize+testSize then break end
    if i%100 == 0 then print("read "..i) end
end
dataTensor = torch.Tensor(#dataTb,1,28,28)
labelTensor = torch.Tensor(#dataTb)
for i=1,#dataTb do
    labelTensor[i] = tonumber(dataTb[i][1])
    for j=1,28 do
        for k=1,28 do
            dataTensor[{i,1,j,k}] = tonumber(dataTb[i][(j-1)*28+k+1])
        end
    end
    if i%100 == 0 then print("cvt "..i) end
end

labelTensor:add(1)
trsize = trainSize
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
