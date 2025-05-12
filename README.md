How It Works
🟢 TRAINING
•	Runs for 10 epochs (can be changed).
•	Takes the fused input (average of focus1 and focus2 images).
•	Uses a CAViT-IMSFN model that combines CNN + Vision Transformer + Spatial Attention.
•	Loss function: MSELoss()+ SSIM loss, optimized using Adam.
🟠 TESTING (Evaluation)
•	After training finishes, the model is evaluated on the 20% test data.
•	Computes:
o	PSNR (Peak Signal-to-Noise Ratio)
o	SSIM (Structural Similarity Index)
o	Time
📄 Model Saving
model_path = f"trained.pth"
torch.save(model.state_dict(), model_path)
📄 Results Saving
with open("meaures.csv", "w", newline="") as f:
________________________________________
📂 Folder Structure Required
dataset/
├── focus1/   # First set of images (input 1)
├── focus2/   # Second set of images (input 2)
└── target/   # Ground truth fused images
________________________________________
▶️ How to Run Training + Testing
In your terminal or script entry point, just call:
train_dataset("dataset", epochs=200)
Replace "dataset" with your actual dataset folder path.
________________________________________
▶️ How to Run Testing
In your terminal or script entry point, just call:
test_dataset("dataset", model_path="trained.pth")
Replace "dataset" with your actual dataset folder path.
________________________________________

