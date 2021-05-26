import matplotlib.pyplot as plt
import numpy as np
import sys

sys.stdin = open('input.txt', 'r')

loss = []
accuracy = []
epoch = 45

for i in range(494):
    log = input()
    if log[1:6]=='EPOCH':
        if len(log) == 55:
            print(log[26:32], log[48:53])
            loss.append(float(log[26:32]))
            # accuracy.append(float(log[48:53]))
        else:
            print(log[27:33], log[49:54])
            loss.append(float(log[27:33]))
            # accuracy.append(float(log[49:54]))

plt.plot(epoch, max(loss), loss)
plt.title('Loss')
plt.show()
        
