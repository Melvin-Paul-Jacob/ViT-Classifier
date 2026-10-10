from torchvision import transforms
from torchvision.datasets import Imagenette
from torch.utils.data import DataLoader, Dataset

class TrainDataset(Dataset):
    def __init__(self, root_dir, image_size):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.dataset = Imagenette(root=root_dir, split="train", 
                                  download=True, transform=self.transform)
        self.classes = self.dataset.classes
        self.num_classes = len(self.classes)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, label = self.dataset[index]
        return {
            "image": image,
            "label": label,
            "index": index
        }

def out_transforms(image_size):
    return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

class ValidationDataset(Dataset):
    def __init__(self, root_dir, image_size):
        self.transform = out_transforms(image_size)
        self.dataset = Imagenette(root=root_dir, split="val", 
                                  download=True, transform=self.transform)
        self.classes = self.dataset.classes
        self.num_classes = len(self.classes)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, label = self.dataset[index]
        return {
            "image": image,
            "label": label,
            "index": index
        }

class TestDataset(Dataset):
    def __init__(self, root_dir, image_size):
        self.transform = out_transforms(image_size)
        self.dataset = Imagenette(root=root_dir, split="test", 
                                  download=True, transform=self.transform)
        self.classes = self.dataset.classes
        self.num_classes = len(self.classes)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, label = self.dataset[index]
        return {
            "image": image,
            "label" : label,
            "index": index
        }

class ViTDataset:
    def __init__(self, image_size, batch_size):
        self.image_size = image_size
        self.batch_size = batch_size
        self.root_dir = "./data"
        self.traindataset = TrainDataset(self.root_dir, self.image_size)
        # self.testdataset = TestDataset(self.root_dir, self.image_size)
        self.valdataset = ValidationDataset(self.root_dir, self.image_size)
        self.num_classes = self.traindataset.num_classes
        self.classes = self.traindataset.classes
        sample = self.traindataset[0]["image"]
        self.image_shape = tuple(sample.shape)

    def train_loader(self):
        return DataLoader(self.traindataset, batch_size=self.batch_size, shuffle=True)

    def test_loader(self):
            return DataLoader(self.testdataset, batch_size=self.batch_size, shuffle=False)

    def val_loader(self):
            return DataLoader(self.valdataset, batch_size=self.batch_size, shuffle=False)

    def __repr__(self):
        return (
            f"ViTDataset(\n"
            f"  dataset      = Imagenette,\n"
            f"  image_shape  = {self.image_shape},\n"
            f"  batch_size   = {self.batch_size},\n"
            f"  num_classes = {self.num_classes},\n"
            f"  train_size  = {len(self.traindataset)},\n"
            f"  val_size    = {len(self.valdataset)},\n"
            # f"  test_size   = {len(self.testdataset)},\n"
            f"  total_size  = {len(self.traindataset) + len(self.valdataset) }\n" # + len(self.testdataset)
            f" train_transform = {self.traindataset.transform},\n" 
            f" val_transform = {self.valdataset.transform},\n" 
            # f" test_transform = {self.testdataset.transform}\n"
            f")"
        )


if __name__ == "__main__": 
    data = ViTDataset(image_size=224, batch_size=32) 
    train_loader = data.train_loader() 
    val_loader = data.val_loader() 
    test_loader = data.test_loader() 
    batch = next(iter(train_loader)) 
    print("Image shape:", batch["image"].shape) 
    print("Labels shape:", batch["label"].shape) 
    print("Indices:", batch["index"][:5]) 
    print("Number of classes:", 102) 
    print("Train samples:", len(data.traindataset)) 
    print("Validation samples:", len(data.valdataset)) 
    print("Test samples:", len(data.testdataset))  