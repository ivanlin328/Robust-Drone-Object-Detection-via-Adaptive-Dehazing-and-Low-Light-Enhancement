import cv2
import argparse
from ultralytics import YOLO
import torch
import pyiqa              
import os 
import importlib.util
import numpy as np

ZERODCEPP_ROOT = "/Users/ivanlin328/Desktop/UCSD/Fall 2025/ECE 253/ECE 253 Final Project/Zero-DCE_extension/Zero-DCE++"

def apply_clahe(input_img, clipLimit=2.0, tileSize=(8,8)):
    """
    Applies CLAHE to a color image correctly by converting to LAB space
    """
    # Convert BGR image to LAB
    lab = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Create CLAHE object
    # clipLimit: contrast value, tileGridSize: grid size
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileSize)  
    
    # Only apply CLAHE on the L (luminance) channel
    l_clahe = clahe.apply(l)

    # Merge the CLAHE-enhanced L channel back with original A and B channels
    lab_clahe = cv2.merge((l_clahe, a, b))
    
    # Convert back from LAB to BGR
    clahe_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    return clahe_img

def load_zerodcepp_module(root):
    """
    Load Zero_DCE++ model
    """
    model_path = os.path.join(root, "model.py")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Can't find model.py：{model_path}")

    spec = importlib.util.spec_from_file_location("zerodcepp_model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_zero_dcepp(device,
                    weight_path=os.path.join(ZERODCEPP_ROOT, "snapshots_Zero_DCE++", "Epoch99.pth"),
                    scale_factor=8):
    """
    Model Initialization & Weight Loading
    """
    zerodcepp_model = load_zerodcepp_module(ZERODCEPP_ROOT)
    # Build model
    net = zerodcepp_model.enhance_net_nopool(scale_factor).to(device)
    # Read weight file
    checkpoint = torch.load(weight_path, map_location=device)
    # Load weights into the model
    net.load_state_dict(checkpoint)
    # Evaluation Mode
    net.eval()
    return net

def apply_zero_dcepp(net, input_img, device,base =32):
    """
    Perform Zero-DCE++ enhancement on the BGR image, and then upload the enhanced BGR image back.
    """
    orig_h, orig_w, _ = input_img.shape
    pad_h = (base - (orig_h % base)) % base
    pad_w = (base - (orig_w % base)) % base

    top, bottom = 0, pad_h
    left, right = 0, pad_w

    img_padded = cv2.copyMakeBorder(
        input_img,
        top, bottom, left, right,
        borderType=cv2.BORDER_REFLECT_101  
    )

    pad_h_img, pad_w_img, _ = img_padded.shape
    # Convert img(BGR->RGB) and Normalization
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    # Transform to Tensor and adjust dimensions:
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device) #unsqueeze(0): Add a batch dimension. Shape becomes (1, C, H, W)
    with torch.no_grad():
        enhanced, _ = net(img_tensor)
    # Convert Tensor back to OpenCV image
    enhanced = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy() #squeeze(0):Remove the batch dimension. Shape: (1, C, H, W) -> (C, H, W)
    enhanced = np.clip(enhanced * 255.0, 0, 255).astype("uint8")
    # Convert color from RGB back to BGR
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
    return enhanced_bgr

def detect_and_save(model, img_path, out_path):
    """
    Runs YOLO detection on an image and saves the result.
    """
    # Run the model
    results = model(img_path, save=True, project=out_path, name="",classes=[0,2])
    print(f"[YOLO] Saved detection results to: {out_path}")
    return results

def compute_image_quality(image_path, device):
    """
    Calculates NIQE and BRISQUE scores for an image.
    """
    if device.type == "mps":
        metric_device = torch.device("cpu")
    else:
        metric_device = device
    # Create metric evaluators 
    niqe_metric = pyiqa.create_metric('niqe', device=metric_device, as_loss=False)
    brisque_metric = pyiqa.create_metric('brisque', device=metric_device, as_loss=False)

    # Load and prepare the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found in compute_image_quality: {image_path}")
        
    # Convert BGR (OpenCV default) to RGB (pyiqa/torch default)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert NumPy (H, W, C) [0-255] to PyTorch Tensor (B, C, H, W) [0-1]
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device) # .unsqueeze(0) adds the Batch dimension

    # Calculate scores
    with torch.no_grad(): 
        niqe_score = niqe_metric(img_tensor).item()
        brisque_score = brisque_metric(img_tensor).item()

    return niqe_score, brisque_score


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Apply CLAHE to an image and run YOLO detection.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input image")
    parser.add_argument("-c", "--clip", type=float, default=2.0, help="CLAHE clip limit")
    parser.add_argument("-t", "--tile", nargs=2, type=int, default=(8,8), help="CLAHE tile size (e.g.: 8 8)")

    args = parser.parse_args()

    input_path = args.input
    clipLimit = args.clip
    tileSize = tuple(args.tile)
    
    # --- Device Setup ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Image ---
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("ERROR: Image not found or could not be read.")

    # --- Load YOLO Model ---
    print("Loading YOLOv8l model...")
    model = YOLO("yolov8l.pt")
    model.to(device)
    
    # Process Original Image
    print("\n--- Running YOLO on [Original Image] ---")
    detect_and_save(model, input_path, "detect_original")
    
    print("\n--- Calculating quality scores for [Original Image] ---")
    try:
        niqe_orig, brisque_orig = compute_image_quality(input_path, device)
        print(f"Original Image NIQE: {niqe_orig:.4f}")
        print(f"Original Image BRISQUE: {brisque_orig:.4f}")
    except Exception as e:
        print(f"Error calculating original image quality: {e}")

    # Process CLAHE Image
    print("\n--- Applying CLAHE ---")
    clahe_img = apply_clahe(img, clipLimit, tileSize)
    clahe_path = "done6_clahe_output.png"
    cv2.imwrite(clahe_path, clahe_img)
    print(f"Saved CLAHE image to: {clahe_path}")
    
    # Process Zero-DCE++ Image
    print("\n--Applying Zero DCE++")
    zero_dcepp_net = load_zero_dcepp(device)
    zd_img = apply_zero_dcepp(zero_dcepp_net, img, device)
    zd_path = "drone6_zerodcepp_output.png"
    cv2.imwrite(zd_path, zd_img)
    print(f"Saved Zero-DCE++ image to: {zd_path}")
    
    
    # Calculate quality scores
    print("\n--- Calculating quality scores for [CLAHE Image] ---")
    try:
        niqe_cl, brisque_cl = compute_image_quality(clahe_path, device)
        print(f"CLAHE Image NIQE: {niqe_cl:.4f}")
        print(f"CLAHE Image BRISQUE: {brisque_cl:.4f}")
        niqe_zd,brisque_zd = compute_image_quality(zd_path, device)
        print(f"Zero DCE++ Image NIQE: {niqe_zd:.4f}")
        print(f"Zero DCE++ Image BRISQUE: {brisque_zd:.4f}")
    except Exception as e:
        print(f"Error calculating CLAHE image quality: {e}")
    
    # Run YOLO on CLAHE image
    print("\n--- Running YOLO on [CLAHE Image] ---")
    detect_and_save(model, clahe_path, "detect_clahe")
    print("\n--- Running YOLO on [Zero DCE++ Image] ---")
    detect_and_save(model, zd_path, "detect_zerodce")