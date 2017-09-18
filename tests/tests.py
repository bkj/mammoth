from tqdm import tqdm
from time import time

n_iter = 2 ** 11
s = np.random.uniform(-1, 1, n_iter)
s = torch.DoubleTensor(s)

# --
# Regular multiplication -- underflows

z = (torch.zeros((1000000, 1)) + 1).cuda().double()
o = z.clone()
c = (torch.zeros((1000000, 1)) + 0.5).cuda().double()

for i in range(n_iter):
    z *= c
    z -= s[i]

for i in range(n_iter)[::-1]:
    z += s[i]
    z /= c

(z == o).all()
z.max()

# --
# ETensor multiplication -- seems to work

z = (torch.zeros((1000000, 1)) + 1).cuda().double()
o = z.clone()
z = ETensor(z)

c = (torch.zeros((1000000, 1)) + 0.5).cuda().double()

t = time()
for i in range(n_iter):
    _ = z.mul(c)
    # _ = z.sub(s.index(i))

print z.val
print len(z.aux_buffer)

for i in range(n_iter)[::-1]:
    # _ = z.add(s.index(i))
    _ = z.unmul(c)

(z.val == o).all()

time() - t
# 4.3 s for 128 steps of 1M params

# --

z = torch.LongTensor([1])

z = z * (2 ** 16)
z


torch.LongTensor([2 ** (63 - 16) * (2 ** 16)])