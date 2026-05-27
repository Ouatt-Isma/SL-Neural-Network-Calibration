import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 30})
# Define a Python script that will read a file containing the provided log data and extract loss and val_loss.
def extract_losses_from_file(filename):
    loss_list = []
    val_loss_list = []

    # Open the file and read its contents
    with open(filename, 'r') as file:
        for line in file:
            # Look for lines that contain 'loss:' and 'val_loss:'
            if 'loss:' in line and 'val_loss:' in line:
                # Extract the loss and val_loss values from the line
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.startswith('loss:'):
                        loss_value = float(parts[i + 1])
                        loss_list.append(loss_value)
                    if part.startswith('val_loss:'):
                        val_loss_value = float(parts[i + 1])
                        val_loss_list.append(val_loss_value)
    
    return loss_list, val_loss_list

# Example usage (assuming the log file is saved as 'training_log.txt')
# loss, val_loss = extract_losses_from_file('mnist_hist.txt')
loss, val_loss = extract_losses_from_file('cifar_hist.txt')
epochs = list(range(1, len(loss)+1))
# Plotting training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss, label='Training Loss', marker='o', color='blue')
plt.plot(epochs, val_loss, label='Validation Loss', marker='o', color='orange')

# Adding title and labels
plt.title('Training Loss vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

# Adding a legend to distinguish between the two lines
plt.legend()

# Showing the plot with grid
plt.grid(True)
plt.show()

