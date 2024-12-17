import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_and_get_complete_report(model, X_test_tensor, y_test_tensor, label_encoder, num_epochs, losses):
    """
    Evaluate the model and generate a complete report including classification report and confusion matrix.

    Parameters:
    model (torch.nn.Module): The trained model to evaluate.
    num_epochs (int): The number of epochs the model was trained for.
    losses (list): A list of loss values recorded during training.

    Returns:
    None
    """
    model.eval()

    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)

    y_test_np = y_test_tensor.numpy()
    predicted_np = predicted.numpy()

    print(classification_report(y_test_np, predicted_np, target_names=label_encoder.classes_, labels=range(len(label_encoder.classes_))))

    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    axs[0].plot(range(1, num_epochs + 1), losses, marker='o')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training Loss Over Epochs')
    axs[0].grid(True)

    conf_matrix = confusion_matrix(y_test_np, predicted_np)
    im = axs[1].imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    axs[1].set_title('Confusion Matrix')

    fig.colorbar(im, ax=axs[1])

    axs[1].set_ylabel('True label')
    axs[1].set_xlabel('Predicted label')

    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['bottom'].set_visible(False)
    axs[1].spines['left'].set_visible(False)

    plt.tight_layout()

    plt.show()
