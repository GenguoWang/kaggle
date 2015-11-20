require 'nn'
require 'torch'
local csv = require 'csv'
local testFile = csv.open("test.csv")
local dataTb = {}
local testSize = 200
local trainSize = 40000
local i=0
for fields in testFile:lines() do
    if i> 0 then
  table.insert(dataTb, fields)
    end
    i=i+1
    if i%100 == 0 then print("tet "..i) end
end
testTensor = torch.Tensor(#dataTb,1,28,28)
for i=1,#dataTb do
    for j=1,28 do
        for k=1,28 do
            testTensor[{i,1,j,k}] = tonumber(dataTb[i][(j-1)*28+k])
        end
    end
    if i%100 == 0 then print("tet "..i) end
end

local trainFile = csv.open("train.csv")
dataTb = {}
i=0
for fields in trainFile:lines() do
    if i> 0 then
          table.insert(dataTb, fields)
    end
    i=i+1
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

obj = {
    testTensor = testTensor,
    dataTensor = dataTensor,
    labelTensor = labelTensor
}
torch.save("data.t7",obj)
