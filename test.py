from matplotlib import pyplot as plt
import numpy as np

trainingEpoch_loss = np.arange(1, 100, 2)
validationEpoch_loss = np.arange(1, 200, 4)

plt.plot(trainingEpoch_loss, label="train_loss")
plt.plot(validationEpoch_loss, label="val_loss")
plt.legend()
plt.show()
