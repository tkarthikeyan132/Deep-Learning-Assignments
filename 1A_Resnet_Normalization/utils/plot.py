import matplotlib.pyplot as plt
import json

def store_result(train_loss_list, val_loss_list, test_loss_list, train_accuracy_list, val_accuracy_list, test_accuracy_list, train_micro_f1_list, val_micro_f1_list, test_micro_f1_list, train_macro_f1_list, val_macro_f1_list, test_macro_f1_list, output_path):
    result = {
        "train":{
            "train_loss": train_loss_list[-1],
            "train_accuracy": train_accuracy_list[-1],
            "train_micro_f1": train_micro_f1_list[-1],
            "train_macro_f1": train_macro_f1_list[-1]
        },
        "val":{
            "val_loss": val_loss_list[-1],
            "val_accuracy": val_accuracy_list[-1],
            "val_micro_f1": val_micro_f1_list[-1],
            "val_macro_f1": val_macro_f1_list[-1]
        },
        "test":{
            "test_loss": test_loss_list[-1],
            "test_accuracy": test_accuracy_list[-1],
            "test_micro_f1": test_micro_f1_list[-1],
            "test_macro_f1": test_macro_f1_list[-1]
        }
    }

    # Store results in a JSON file
    output_file_path = "results.json"
    with open(output_path + output_file_path, "w") as json_file:
        json.dump(result, json_file, indent=4)


# Plotting the graphs
def all_plots(epochs, train_loss_list, val_loss_list, test_loss_list, train_accuracy_list, val_accuracy_list, test_accuracy_list, train_micro_f1_list, val_micro_f1_list, test_micro_f1_list, train_macro_f1_list, val_macro_f1_list, test_macro_f1_list, output_path):
    #Loss vs epoch graph
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss_list, label='Train Loss')
    plt.plot(epochs, val_loss_list, label='Validation Loss')
    plt.plot(epochs, test_loss_list, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    plt.savefig(output_path + 'loss_plot.png')
    # plt.show()

    #Accuracy vs epoch graph
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracy_list, label='Train Accuracy')
    plt.plot(epochs, val_accuracy_list, label='Validation Accuracy')
    plt.plot(epochs, test_accuracy_list, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.legend()
    plt.savefig(output_path + 'accuracy_plot.png')
    # plt.show()

    #Micro F1 vs epoch graph
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_micro_f1_list, label='Train Micro F1')
    plt.plot(epochs, val_micro_f1_list, label='Validation Micro F1')
    plt.plot(epochs, test_micro_f1_list, label='Test Micro F1')
    plt.xlabel('Epochs')
    plt.ylabel('Micro F1')
    plt.title('Micro F1 vs Epochs')
    plt.legend()
    plt.savefig(output_path + 'micro_f1_plot.png')
    # plt.show()

    #Macro F1 vs epoch graph
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_macro_f1_list, label='Train Macro F1')
    plt.plot(epochs, val_macro_f1_list, label='Validation Macro F1')
    plt.plot(epochs, test_macro_f1_list, label='Test Macro F1')
    plt.xlabel('Epochs')
    plt.ylabel('Macro F1')
    plt.title('Macro F1 vs Epochs')
    plt.legend()
    plt.savefig(output_path + 'macro_f1_plot.png')
    # plt.show()

    store_result(train_loss_list, val_loss_list, test_loss_list, train_accuracy_list, val_accuracy_list, test_accuracy_list, train_micro_f1_list, val_micro_f1_list, test_micro_f1_list, train_macro_f1_list, val_macro_f1_list, test_macro_f1_list, output_path)