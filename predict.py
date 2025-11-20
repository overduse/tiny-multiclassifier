import torch
from torchvision import transforms
from PIL import Image
import os

# import your model
from model import SimpleCNN, BNCNN

MODEL_WEIGHTS_PATH = './checkpoints/best_model.pth'

MEAN_VAL = 0.8435
STD_VAL = 0.2694

DATA_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((MEAN_VAL,), (STD_VAL,))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BNCNN().to(device)
# model = SimpleCNN().to(device)

try:
    map_location=device
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
except FileNotFoundError:
    print(f"Error: can not find the model weight file '{MODEL_WEIGHTS_PATH}'")
    print("please run python train.py firstly to attain the model weight file.")
    exit()

model.eval()
print(f"model '{MODEL_WEIGHTS_PATH}' successfully loaded, running on {device}.")


def predict(img_path: str) -> int:
    """
    lodal model and take a single image as input, and
    outputs the result

    Args:
        img_path (str): input image path

    Returns:
        int: output of the function, label (0-9)
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"can not find the image file: {img_path}")

    try:
        # load and preprocess
        # using .convert('L') to make sure the image is gray-scale
        image = Image.open(img_path).convert('L')
        image_tensor = DATA_TRANSFORM(image)
        
        # expand the batch dim, [batch_size, channels, height, width]
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted_index = torch.max(outputs, 1)
            
        return predicted_index.item()

    except Exception as e:
        print(f"something wrong: {e}")
        return -1


if __name__ == '__main__':
    # Testing Block

    TEST_IMAGE_PATH = './data/train/3/1155109903-IMG_3_2_1155109903.png'
    # TEST_IMAGE_PATH = './data/train/7/1155255463-IMG_7_4_1155255463.png'
    # TEST_IMAGE_PATH = './data/train/10/1155252314-IMG_10_8_1155252314.png'

    print(f"Running prediction for a single test image: {TEST_IMAGE_PATH}")

    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Error: Test image not found at '{TEST_IMAGE_PATH}'")
        print("Please update the TEST_IMAGE_PATH variable in the script.")
    else:
        predicted_label = predict(TEST_IMAGE_PATH)
        
        if predicted_label != -1:
            label_map = {
                0: "香", 1: "港", 2: "中", 3: "文", 4: "大",
                5: "學", 6: "人", 7: "工", 8: "智", 9: "能"
            }
            predicted_word = label_map.get(predicted_label, "Unknown")

            print("\n--- Prediction Result ---")
            print(f"Input Image:    {TEST_IMAGE_PATH}")
            print(f"Predicted Label:  {predicted_label} (Corresponds to: '{predicted_word}')")