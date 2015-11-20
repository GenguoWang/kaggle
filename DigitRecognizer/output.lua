require 'torch'
print("ImageId,Label")
i=1
label = torch.load("results/label.t7")
label:apply(function(x) print(i..","..x);i=i+1 end)
