import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load data from the outputs.pkl file
with open('outputs_42454.pkl', 'rb') as file:
    outputs = pickle.load(file)

print(outputs)

# Count number of epochs
epochs = range(1, len(outputs[0][1]) + 1)

# Generate line graphs for each tuple in outputs
for i, data_tuple in enumerate(outputs):
    mislabel_ratio, loss_values, accuracy_values, val_loss_values, val_accuracy_values = data_tuple

    # Loss graph
    plt.figure()
    plt.plot(epochs, loss_values, label='Train Loss')
    plt.plot(epochs, val_loss_values, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.title(f'Epochs vs Loss (Mislabel Ratio: {mislabel_ratio * 100:.2f}%)')
    plt.legend()
    plt.xticks(np.arange(min(epochs), max(epochs) + 1, 1))
    plt.show()

    # Accuracy graph
    plt.figure()
    plt.plot(epochs, accuracy_values, label='Train Accuracy')
    plt.plot(epochs, val_accuracy_values, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.title(f'Epochs vs Accuracy (Mislabel Ratio: {mislabel_ratio * 100:.2f}%)')
    plt.legend()
    plt.xticks(np.arange(min(epochs), max(epochs) + 1, 1))
    plt.show()

# Extract the values for averaging
loss_values_avg = np.mean([data_tuple[1] for data_tuple in outputs], axis=0)
accuracy_values_avg = np.mean([data_tuple[2] for data_tuple in outputs], axis=0)
val_loss_values_avg = np.mean([data_tuple[3] for data_tuple in outputs], axis=0)
val_accuracy_values_avg = np.mean([data_tuple[4] for data_tuple in outputs], axis=0)

# Generate line graph for average loss
plt.figure()
plt.plot(epochs, loss_values_avg, label='Average Train Loss')
plt.plot(epochs, val_loss_values_avg, label='Average Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0, 1)
plt.title('Epochs vs Average Loss')
plt.legend()
plt.xticks(np.arange(min(epochs), max(epochs)+1, 1))
plt.show()

# Generate line graph for average accuracy
plt.figure()
plt.plot(epochs, accuracy_values_avg, label='Average Train Accuracy')
plt.plot(epochs, val_accuracy_values_avg, label='Average Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.title('Epochs vs Average Accuracy')
plt.legend()
plt.xticks(np.arange(min(epochs), max(epochs)+1, 1))
plt.show()
