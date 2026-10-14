import json
import torch
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from config import vit_config
from dataset import out_transforms, ViTDataset
from model import VisionTransformer


# --------------------------------------------------
# 4. Image prediction function
# --------------------------------------------------

def predict(image_path, model, classes):
    """
    Predict the class of an image using a trained ViT model.
    Returns the predicted class and confidence.
    """

    try:
        # Use the same image size and normalization as validation.
        transform = out_transforms(vit_config.image_size)

        # Load image
        img = Image.open(image_path).convert("RGB")

        # Transform image and add batch dimension
        image = transform(img).unsqueeze(0).to(vit_config.device)

        print(f"Transformed image shape: {image.shape}")

        # Run inference
        with torch.no_grad():
            outputs = model(image)
            probabilities = F.softmax(outputs, dim=1)

        # Get predicted class
        pred_index = probabilities.argmax(dim=1).item()

        pred_class = classes[pred_index]
        confidence = probabilities[0, pred_index].item()

        # Get top 5 predictions
        top_k = min(5, len(classes))

        top_probs, top_indices = torch.topk(
            probabilities[0],
            k=top_k,
        )

        top_classes = [
            classes[index]
            for index in top_indices.cpu().tolist()
        ]

        top_probs = top_probs.cpu().tolist()

        # --------------------------------------------------
        # 5. Display image and predictions
        # --------------------------------------------------

        # Reverse ImageNet normalization for visualization.
        mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(3, 1, 1)

        display_image = image[0].detach().cpu() * std + mean
        display_image = display_image.clamp(0, 1)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        axes[0].imshow(display_image.permute(1, 2, 0).numpy())
        axes[0].set_title(
            f"Prediction: {pred_class}\n"
            f"Confidence: {confidence:.2%}"
        )
        axes[0].axis("off")

        axes[1].barh(
            top_classes[::-1],
            top_probs[::-1],
        )
        axes[1].set_xlabel("Probability")
        axes[1].set_title(f"Top {top_k} Class Probabilities")
        axes[1].set_xlim(0, 1)

        plt.tight_layout()
        plt.show()

        return pred_class, confidence

    except Exception as e:
        raise RuntimeError(
            f"Failed to predict image '{image_path}': {e}"
        ) from e

def predict_validation_samples(
    model,
    dataset,
    classes,
    device,
    num_samples=8,
):
    model.eval()

    # Select random validation indices
    num_samples = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)

    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    # Display images in a grid
    fig, axes = plt.subplots(
        num_samples, 2,
        figsize=(13, 4 * num_samples),
        squeeze=False,
    )

    correct = 0

    for row, index in enumerate(indices):
        sample = dataset[index]

        image = sample["image"]
        true_label = sample["label"]

        # Add batch dimension and move to device
        input_tensor = image.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)

        pred_index = probabilities.argmax(dim=1).item()
        confidence = probabilities[0, pred_index].item()

        true_class = classes[true_label]
        pred_class = classes[pred_index]

        correct += int(pred_index == true_label)

        # Undo normalization for display
        display_image = (
            image.cpu() * std + mean
        ).clamp(0, 1)

        axes[row, 0].imshow(
            display_image.permute(1, 2, 0).numpy()
        )

        status = "CORRECT" if pred_index == true_label else "INCORRECT"

        axes[row, 0].set_title(
            f"True: {true_class}\n"
            f"Predicted: {pred_class}\n"
            f"Confidence: {confidence:.2%} | {status}"
        )
        axes[row, 0].axis("off")

        # Top-5 predictions
        k = min(5, len(classes))
        top_probs, top_indices = torch.topk(
            probabilities[0], k=k
        )

        top_classes = [
            classes[i]
            for i in top_indices.cpu().tolist()
        ]

        axes[row, 1].barh(
            top_classes[::-1],
            top_probs.cpu().tolist()[::-1],
        )
        axes[row, 1].set_title("Top-5 Predictions")
        axes[row, 1].set_xlabel("Probability")
        axes[row, 1].set_xlim(0, 1)

        print(
            f"Index: {index} | "
            f"True: {true_class} | "
            f"Predicted: {pred_class} | "
            f"Confidence: {confidence:.2%}"
        )

    accuracy = correct / num_samples

    print(
        f"\nAccuracy on these {num_samples} sampled images: "
        f"{accuracy:.2%}"
    )

    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# 6. Run prediction
# --------------------------------------------------

if __name__ == "__main__":
    # --------------------------------------------------
    # 1. Load checkpoint
    # --------------------------------------------------

    checkpoint_path = "outputs/checkpoints/best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=vit_config.device, 
                            weights_only=False)
    inference_model = VisionTransformer(vit_config).to(vit_config.device)
    inference_model.load_state_dict(checkpoint["model_state_dict"])
    inference_model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Validation accuracy: {checkpoint['val_accuracy']:.4f}")

    # --------------------------------------------------
    # 2. Load training history (optional)
    # --------------------------------------------------
    history_path = "outputs/history/train_history.json"
    try:
        with open(history_path, "r", encoding="utf-8") as file:
            train_history = json.load(file)

        print("Training history loaded successfully.")
    except FileNotFoundError:
        train_history = None
        print("Training history JSON file not found.")


    # --------------------------------------------------
    # 3. Load class names from the dataset
    # --------------------------------------------------
    # Load the validation dataset
    data = ViTDataset(
        image_size=vit_config.image_size,
        batch_size=32,
    )
    classes = [
        c[0] if isinstance(c, (tuple, list)) else c
        for c in data.classes
    ]
    print(f"Number of classes: {len(classes)}")
    print(f"Classes: {classes}")

    predict_validation_samples(
        model=inference_model,
        dataset=data.valdataset,
        classes=classes,
        device=vit_config.device,
        num_samples=8,
    )
