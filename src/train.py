import json
import numpy 
import torch
import random
import timeit
from torch import nn
from tqdm import tqdm
from torch import optim
from pathlib import Path
from torchinfo import summary
from config import vit_config
from dataset import ViTDataset
import matplotlib.pyplot as plt
from model import VisionTransformer
from torch.optim.lr_scheduler import CosineAnnealingLR

config = vit_config

random.seed(config.random_seed)
numpy.random.seed(config.random_seed)
torch.manual_seed(config.random_seed)
torch.cuda.manual_seed(config.random_seed)
torch.cuda.manual_seed_all(config.random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

dataset = ViTDataset(config.image_size, config.batch_size)
print(dataset)
train_dataloader = dataset.train_loader()
val_dataloader = dataset.val_loader()
# test_dataloader = dataset.test_loader()

classes = dataset.classes  
num_classes = dataset.num_classes  
image_shape = dataset.image_shape

# ImageNet normalization parameters 
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1) 
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) 
def denormalize(image): 
    image = image.cpu() * std + mean 
    return image.clamp(0, 1).permute(1, 2, 0).numpy() 
# Plot sample images 
fig, axarr = plt.subplots(1, 3, figsize=(15, 5)) 
# Training image 
img = dataset.traindataset[0]["image"] 
axarr[0].imshow(denormalize(img)) 
axarr[0].set_title("Train Image") 
axarr[0].axis("off")
# Validation image 
# img1 = dataset.testdataset[10]["image"] 
# axarr[1].imshow(denormalize(img1)) 
# axarr[1].set_title("Test Image") 
# axarr[1].axis("off") 
# Test image 
img2 = dataset.valdataset[6]["image"] 
axarr[2].imshow(denormalize(img2)) 
axarr[2].set_title("Val Image") 
axarr[2].axis("off") 
plt.tight_layout() 
plt.show()


config.num_classes = num_classes
model = VisionTransformer(config).to(config.device)

summary(model=model, 
        input_size=(dataset.batch_size, image_shape[0], image_shape[1], image_shape[2]), 
        col_names= ["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings= ["var_names"]
    )


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), betas=config.adam_betas, 
                       lr=config.learning_rate, weight_decay=config.adam_weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

# ------------------------------------------------------------------
# Output directories
# ------------------------------------------------------------------
output_dir = Path("outputs")
checkpoint_dir = output_dir / "checkpoints"
history_dir = output_dir / "history"
plot_dir = output_dir / "plots"

checkpoint_dir.mkdir(parents=True, exist_ok=True)
history_dir.mkdir(parents=True, exist_ok=True)
plot_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Training history
# ------------------------------------------------------------------
train_history = {
    "epoch": [],
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
    "learning_rate": [],
}

best_val_accuracy = float("-inf")

start = timeit.default_timer()

for epoch in range(config.epochs):
    print(f"Epoch: {epoch+1}/{config.epochs}")
    # training
    model.train()
    train_running_loss = 0
    train_correct = 0 
    train_total = 0

    train_bar = tqdm(train_dataloader, desc="Train", position=0, leave=True)
    
    for idx, image in enumerate(train_bar):
        img = image["image"].float().to(config.device)
        label = image["label"].type(torch.uint8).to(config.device)
        
        y_pred = model(img)
        loss = criterion(y_pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = label.size(0)
        train_total += batch_size
        # Running loss and accuracy
        train_running_loss += loss.item() * batch_size
        preds = y_pred.argmax(dim=1) 
        train_correct += (preds == label).sum().item() 

        current_train_loss = train_running_loss / train_total
        current_train_accuracy = train_correct / train_total
        
        train_bar.set_postfix(loss=f"{current_train_loss:.4f}", 
                              acc=f"{current_train_accuracy:.4f}", 
                              lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    train_loss = train_running_loss / train_total
    train_accuracy = train_correct / train_total
    
    #validation
    model.eval()
    valid_running_loss = 0
    valid_correct = 0 
    valid_total = 0

    val_bar = tqdm(val_dataloader, desc="Valid", position=0, leave=True)

    with torch.no_grad():
        for idx, image in enumerate(val_bar):
            img = image["image"].float().to(config.device)
            label = image["label"].type(torch.uint8).to(config.device)
            
            y_pred = model(img)
            loss = criterion(y_pred, label)

            batch_size = label.size(0)

            valid_running_loss += loss.item() * batch_size
            valid_total += batch_size

            preds = y_pred.argmax(dim=1) 
            valid_correct += (preds == label).sum().item() 

            current_val_loss = valid_running_loss / valid_total
            current_val_accuracy = valid_correct / valid_total

            val_bar.set_postfix(loss=f"{current_val_loss:.4f}", 
                                acc=f"{current_val_accuracy:.4f}")

    val_loss = valid_running_loss / valid_total
    val_accuracy = valid_correct / valid_total

    current_lr = optimizer.param_groups[0]["lr"]

    # ==============================================================
    # Store epoch history
    # ==============================================================
    train_history["epoch"].append(epoch + 1)
    train_history["train_loss"].append(train_loss)
    train_history["train_accuracy"].append(train_accuracy)
    train_history["val_loss"].append(val_loss)
    train_history["val_accuracy"].append(val_accuracy)
    train_history["learning_rate"].append(current_lr)

    # ==============================================================
    # Save best model
    # ==============================================================
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        best_checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "best_val_accuracy": best_val_accuracy,
            "classes": classes,
            "num_classes": num_classes,
            "image_shape": image_shape,
        }

        torch.save(
            best_checkpoint,
            checkpoint_dir / "best_model.pt",
        )

        tqdm.write(
            f"Saved best model: validation accuracy "
            f"{best_val_accuracy:.4f}"
        )

    # Save history after every epoch so it survives interruptions.
    with open(history_dir / "train_history.json", "w", encoding="utf-8",) as file:
        json.dump(train_history, file, indent=4)

    torch.save(train_history, history_dir / "train_history.pt")
    
    scheduler.step()
    # Final epoch summary in tqdm 
    tqdm.write( f"Epoch {epoch + 1}/{config.epochs} | " 
               f"Train Loss: {train_loss:.4f} | " 
               f"Train Acc: {train_accuracy:.4f} | " 
               f"Val Loss: {val_loss:.4f} | " 
               f"Val Acc: {val_accuracy:.4f} | " 
               f"LR: {optimizer.param_groups[0]['lr']:.2e}" )
        
stop = timeit.default_timer()
training_time = stop - start
print(f"Training Time: {training_time:.2f}s")
print(f"Best validation accuracy: {best_val_accuracy:.4f}")

final_checkpoint = {
    "epoch": config.epochs,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "best_val_accuracy": best_val_accuracy,
    "training_time_seconds": training_time,
    "classes": classes,
    "num_classes": num_classes,
    "image_shape": image_shape,
    "train_history": train_history,
}

torch.save(
    final_checkpoint,
    checkpoint_dir / "final_model.pt",
)

print(f"Best model saved to: {checkpoint_dir / 'best_model.pt'}")
print(f"Final model saved to: {checkpoint_dir / 'final_model.pt'}")
print(f"History saved to: {history_dir / 'train_history.json'}")

epochs = train_history["epoch"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss
axes[0].plot(
    epochs,
    train_history["train_loss"],
    marker="o",
    label="Training loss",
)
axes[0].plot(
    epochs,
    train_history["val_loss"],
    marker="o",
    label="Validation loss",
)
axes[0].set_title("Training and Validation Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-entropy loss")
axes[0].grid(alpha=0.3)
axes[0].legend()

# Accuracy
axes[1].plot(
    epochs,
    train_history["train_accuracy"],
    marker="o",
    label="Training accuracy",
)
axes[1].plot(
    epochs,
    train_history["val_accuracy"],
    marker="o",
    label="Validation accuracy",
)
axes[1].set_title("Training and Validation Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].grid(alpha=0.3)
axes[1].legend()

# Learning rate
axes[2].plot(
    epochs,
    train_history["learning_rate"],
    marker="o",
    color="tab:green",
)
axes[2].set_title("Learning Rate")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Learning rate")
axes[2].ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
axes[2].grid(alpha=0.3)

plt.tight_layout()

history_plot_path = plot_dir / "train_history.png"
plt.savefig(
    history_plot_path,
    dpi=200,
    bbox_inches="tight",
)
plt.show()

print(f"Training plot saved to: {history_plot_path}")

# test_images = []
# test_preds = []
# test_labels = []

# model.eval()
# with torch.no_grad():
#     for idx, image in enumerate(tqdm(test_dataloader, position=0, desc="Testing")):
#         img = image["image"].to(config.device)
#         label = image["label"].to(config.device)
        
#         y_pred = model(img)
#         y_pred_label = torch.argmax(y_pred, dim=1)
        
#         test_images.extend(img.cpu().detach())
#         test_preds.extend([int(i) for i in y_pred_label])
#         test_labels.extend(label.cpu().detach())

#     print(f"Test accuracy : {sum(1 for x, y in zip(test_labels, test_preds) if x==y)/len(test_labels):.4f}")


# plt.figure(figsize=(8, 5))  # Larger figure for better visibility
# f, axarr = plt.subplots(2, 3, figsize=( 8, 5))
# counter = 0

# for i in range(2):
#     for j in range(3):
#         # Display the image
#         axarr[i][j].imshow(test_images[counter].squeeze().permute(1, 2, 0))
        
#         # Get predicted and actual labels
#         pred_label = classes[test_preds[counter]]
#         actual_label = classes[test_labels[counter]]  # Assuming test_labels contains the true labels
        
#         # Set title to show both predicted and actual labels
#         axarr[i][j].set_title(f"Pred: {pred_label}\nActual: {actual_label}", fontsize=12)
        
#         # Remove axis ticks for cleaner display
#         axarr[i][j].set_xticks([])
#         axarr[i][j].set_yticks([])
        
#         counter += 1

# plt.tight_layout(pad=3.0)  # Add padding between subplots
# plt.suptitle("Model Predictions vs Actual Labels", fontsize=16, y=0.98)
# plt.show()

# torch.save(model, 'vision_transformer_flowers_full.pth')

