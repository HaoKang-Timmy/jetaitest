# load /home/haokang/jetaitest/test/Ai_20251030062508.pt and/home/haokang/jetaitest/test/Ai_20251030062712.pt

import torch

ai1 = torch.load("/home/haokang/jetaitest/test/Ai_20251030062508.pt")
ai2 = torch.load("/home/haokang/jetaitest/test/Ai_20251030062712.pt")

# 打印两个tensor，看看差在哪儿？
print("ai1:")
print(ai1)
print("ai2:")
print(ai2)
print("ai1 and ai2 are equal:")
# Skip locations where either ai1 or ai2 is nan when comparing
import torch

# Get mask of floats which are NOT nan (for both tensors)
mask = (~torch.isnan(ai1)) & (~torch.isnan(ai2))

# Check for close values where both are not nan
torch.testing.assert_close(ai1[mask], ai2[mask])

print("ai1 and ai2 are equal (ignoring NaNs)")

# For strict equality ignoring nans:
assert torch.equal(ai1[mask], ai2[mask])
print("ai1 and ai2 are equal")