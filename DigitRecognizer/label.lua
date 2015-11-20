require 'data'
model = torch.load("results/model.net")
length = testTensor:size(1)
resultTensor = torch.Tensor(length)
batchSize = 128
for i=0,math.floor((length-1)/batchSize) do
    l = math.min(batchSize,length-i*batchSize)
    output = model:forward(testTensor[{{i*batchSize+1,i*batchSize+l}}])
    m,index = torch.max(output,2)
    index:add(-1)
    resultTensor[{{i*batchSize+1,i*batchSize+l}}] = index:squeeze()
end
torch.save("results/label.t7",resultTensor)
