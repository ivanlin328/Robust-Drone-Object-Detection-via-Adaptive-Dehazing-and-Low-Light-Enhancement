import cv2
import math
import numpy as np
import torch
from lowlight import detect_and_save,compute_image_quality
from ultralytics import YOLO


def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz,3)

    indices = darkvec.argsort()
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.95
    im3 = np.empty(im.shape,im.dtype)

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz)
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
    cov_Ip = mean_Ip - mean_I*mean_p

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r))
    var_I   = mean_II - mean_I*mean_I

    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))

    q = mean_a*im + mean_b
    return q

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray,et,r,eps)

    return t

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype)
    t = cv2.max(t,tx)

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res

def detect_and_save(model, img_path, out_path):
    """
    Runs YOLO detection on an image and saves the result.
    """
    # Run the model
    results = model(img_path, save=True, project=out_path, name="",classes=[0,2])
    print(f"[YOLO] Saved detection results to: {out_path}")
    return results

if __name__ == '__main__':

    # --- Device Setup ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    # Load Original Image
    FILEPATH = '/Users/ivanlin328/Desktop/UCSD/Fall 2025/ECE 253/ECE 253 Final Project/dataset_original_fog/images/val/fog10.jpg'
    img = cv2.imread(FILEPATH)
    # Calculate original images quality scores
    niqe_or, brisque_or = compute_image_quality(FILEPATH, device)
    print(f"original Image NIQE: {niqe_or:.4f}")
    print(f"original Image BRISQUE: {brisque_or:.4f}")
    
    
    # --- Load YOLO Model ---
    print("Loading YOLOv8l model...")
    model = YOLO("yolov8l.pt")
    model.to(device)
    
    # Process Original Image
    print("\n--- Running YOLO on [Original Image] ---")
    detect_and_save(model, FILEPATH, "detect_original_fog")
    
    
    # Process DCP Image
    I = img.astype('float64')/255
    dark = DarkChannel(I,15)
    A = AtmLight(I,dark)
    te = TransmissionEstimate(I,A,15)
    t = TransmissionRefine(img,te)
    J = Recover(I,t,A,0.1)
    dcp_path = "/Users/ivanlin328/Desktop/UCSD/Fall 2025/ECE 253/ECE 253 Final Project/dataset_dcp/images/val/dcp_fog10.jpg"
    cv2.imwrite(dcp_path,J*255)
    
    # Calculate DCP images quality scores
    print("\n--- Calculating DCP images quality scores ---")
    niqe_cl, brisque_cl = compute_image_quality(dcp_path, device)
    print(f"DCP Image NIQE: {niqe_cl:.4f}")
    print(f"DCP Image BRISQUE: {brisque_cl:.4f}")
    
    # Run YOLO on CLAHE image
    print("\n--- Running YOLO on [DCP Image] ---")
    detect_and_save(model,dcp_path , "detect_dcp")
    
    
        

    