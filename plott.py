import numpy as np
import matplotlib.pyplot as plt

loss=np.array([126,34,70,50,45,33,17,14,18,13.4,12,11.5,11,10,8.92,5.00,3.70,4.22,3.5,3.33,3.5,5,3.33,3.2,3.18,3.15,2.87,2.45,2.90,2.15,2.00,1.78,1.69,1.60,1.50,1.44,1.33,1.69,1.25,1.18,1.16,1.20,1.19,1.15,1.14,1.10,1.00,0.986,0.96,0.75,0.88,0.75,0.66,0.58,0.54,0.52,0.50,0.49,0.48,0.45,0.44,0.44,0.42,0.42,0.41,0.33,0.33,0.33,0.33,0.33])
print(len(loss))
acc=np.array([6.7,19.4,29.18,33,35,40.59,42.36,44,43,45,48,49.5,50.2,51,52.5,53,57,58,58.6,59,59.25,60,61,63,63.8,64,65,65.42,68.76,69,71,72,74.6,77,77.5,77,77.8,78,78.35,78.59,77,76,76,76,77,78,78.5,79,79.5,79,79,80.2,80.1,80.5,81,81.2,82,81,82,83.3,84,85,85.5,86,87,88.4,88.4,88.4,88.4,88.4])
print(len(acc))

epochs=[]
for i in range (70):
 epochs.append(i)
    
 
epochs=np.array(epochs)
plt.plot(epochs, loss, 'g', label='Training loss',linewidth=2.0)
plt.plot(epochs, acc, 'b', label='Training Accuracy',linewidth=2.0)
plt.title('Training loss & Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()