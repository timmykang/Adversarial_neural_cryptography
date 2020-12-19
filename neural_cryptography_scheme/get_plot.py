from matplotlib import pyplot as plt

step = []
Eve_bits = []
for i in range(1,12):
    step.append(i * 500)

with open('Testing_Eve1.txt', 'r') as f:
    for i in range(11):
        Eve_bits.append(float(f.readline().split()[-1]))


plt.plot(step, Eve_bits)

plt.xlabel('Step')

plt.ylabel('Bit Accuracy')
plt.ylim([0, 16])
plt.title('Result')
plt.legend(['Eve1_0'])
plt.show()