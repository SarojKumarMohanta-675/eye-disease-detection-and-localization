from models.cnn_dataloader import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders()

print("Train size:", len(train_loader.dataset))
print("Validation size:", len(val_loader.dataset))
print("Test size:", len(test_loader.dataset))