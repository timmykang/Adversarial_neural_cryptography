from matplotlib import pyplot as plt

step = []
Eve_bits = []
Bob_bits = []

with open('result0.txt', 'r') as f:
    while(True):
        try:
            step.append(int(f.readline().split()[-1]))
            Eve_bits.append(float(f.readline().split()[3][:-1]))
            Bob_bits.append(float(f.readline().split()[3][:-1]))
        except:
            break

plt.plot(step, Eve_bits)
plt.plot(step, Bob_bits)
plt.xlabel('Step')
plt.ylabel('Bit Accuracy')
plt.title('Result')
plt.legend(['Eve', 'Bob'])
plt.show()