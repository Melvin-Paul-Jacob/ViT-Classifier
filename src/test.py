import json
import torch
from PIL import Image
from config import vit_config
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset import out_transforms
from model import VisionTransformer

checkpoint = torch.load(
    "outputs/checkpoints/best_model.pt",
    map_location=vit_config.device,
    weights_only=False,
)

inference_model = VisionTransformer(vit_config).to(vit_config.device)
inference_model.load_state_dict(checkpoint["model_state_dict"])
inference_model.eval()

print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
print(
    f"Validation accuracy: "
    f"{checkpoint['val_accuracy']:.4f}"
)

with open(
    "outputs/history/train_history.json",
    "r",
    encoding="utf-8",
) as file:
    train_history = json.load(file)

train_history = torch.load(
    "outputs/history/train_history.pt",
    weights_only=False,
)    

def predict(image_path, model,classes):
    """
        Predict flower class from an image
    """
    
    try:
        transform = out_transforms(vit_config)
        
        # Load and transform image
        img = Image.open(image_path).convert('RGB')
        image = transform(img).unsqueeze(0).to(vit_config.device)
        
        # Print shape for debugging
        print(f"Transformed image shape: {image.shape}")
        
        with torch.no_grad():
            out = model(image)
            probabilities = F.softmax(out, dim=1)
            pred = torch.argmax(probabilities, dim=1)
            
        pred_class = classes[pred.item()]
        confidence = probabilities[0][pred.item()].item()
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image[0].cpu().permute(1, 2, 0).numpy())
        plt.title(f"Prediction: {pred_class}\nConfidence: {confidence:.4f}")
        plt.axis('off')
            
        # Plot probability distribution
        plt.subplot(1, 2, 2)
        sorted_indices = torch.argsort(probabilities[0], descending=True).cpu().numpy()
        top_classes = [classes[i] for i in sorted_indices[:5]]
        top_probs = [probabilities[0][i].item() for i in sorted_indices[:5]]
        
        plt.bar(range(len(top_classes)), top_probs)
        plt.xticks(range(len(top_classes)), top_classes, rotation=45)
        plt.title("Top 5 Class Probabilities")
        plt.tight_layout()
        plt.show()
        
        return pred_class, confidence
    
    except Exception as e:
        raise e


classes = ['balloon flower', 'camellia', 'clematis','geranium','japanese anemone',
  'magnolia','purple coneflower','spring crocus','stemless gentian','trumpet creeper']

image_path = "inference\image_02647_jpg.rf.ef01491d5e51acb96500738b675b7019.jpg"
class_name, conf = predict(image_path, inference_model, classes)
print(f"Predicted: {class_name} with {conf*100:.2f}% confidence")