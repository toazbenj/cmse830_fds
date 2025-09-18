var = 0
prob = [0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1]
mean = 0

for i in range(3,10):
    print(i)
    mean += prob[i-3] * i
print(mean)

for i in range(3,10):
    var += (i - mean)**2 * prob[i-3]
print(var)