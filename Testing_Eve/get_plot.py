from matplotlib import pyplot as plt

step = []
Eve_bits = []
for i in range(1,11):
    step.append(i * 500)

with open('Testing_Eve.txt', 'r') as f:
    for i in range(5):
        tmp = []
        for j in range(10):
            tmp.append(float(f.readline().split()[-1]))
        Eve_bits.append(tmp)

for i in range(5):
    plt.plot(step, Eve_bits[i])

plt.xlabel('Step')

plt.ylabel('Bit Accuracy')
plt.ylim([0, 16])
plt.title('Result')
plt.legend(['Eve_0', 'Eve_1', 'Eve_2', 'Eve_3', 'Eve_4'])
plt.show()