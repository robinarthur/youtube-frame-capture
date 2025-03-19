import os
import cv2
import numpy as np
import pytesseract
import argparse
import json
from datetime import datetime
from skimage.metrics import structural_similarity

# Set the path to tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_symbol(img_path, region=((125, 145, 59, 90)), save_dir=None, filename=None):
    """Extract stock symbol using OCR from specified region and optionally save the cropped region."""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None, None
            
        # Crop to symbol region (top-Y, bottom-Y, left-X, right-X)
        y1, y2, x1, x2 = region
        symbol_region = img[y1:y2, x1:x2]
        
        # Save the cropped region if save_dir is provided
        if save_dir and filename:
            os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
            cropped_path = os.path.join(save_dir, f"cropped_{filename}")
            cv2.imwrite(cropped_path, symbol_region)
        
        # Enhanced preprocessing pipeline
        # 1. Convert to grayscale
        gray = cv2.cvtColor(symbol_region, cv2.COLOR_BGR2GRAY)
        
        # 2. Invert colors (black background to white)
        inverted = cv2.bitwise_not(gray)
        
        # 3. Scale up image (300% for better OCR accuracy)
        scaled = cv2.resize(inverted, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        # 4. Adaptive thresholding
        thresh = cv2.adaptiveThreshold(scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        
        # 5. Noise reduction
        denoised = cv2.medianBlur(thresh, 3)
        
        # 6. Dilation to enhance text
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.dilate(denoised, kernel, iterations=1)

        # OCR configuration
        custom_config = r'--oem 2 --psm 7 --dpi 300 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        symbol = pytesseract.image_to_string(processed, lang="eng", config=custom_config)
        
        return symbol.strip(), img
        
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return None, None

def compare_regions(img1_path, img2_path, region, threshold=0.9):
    """Compare similarity of regions using SSIM."""
    try:
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        y1, y2, x1, x2 = region
        roi1 = img1[y1:y2, x1:x2]
        roi2 = img2[y1:y2, x1:x2]
        
        gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
        
        score, _ = structural_similarity(gray1, gray2, full=True)
        return score < threshold
        
    except Exception as e:
        print(f"Comparison error: {str(e)}")
        return True

def save_image_with_box(img, region, output_path):
    """Save image with a box around the specified region."""
    y1, y2, x1, x2 = region
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box with thickness of 2
    cv2.imwrite(output_path, img)

def detect_changes(screenshot_dir, output_file, output_image, cropped_dir):
    """Main detection logic."""
    files = sorted([f for f in os.listdir(screenshot_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    changes = []
    previous = None
    
    # (top-Y, bottom-Y, left-X, right-X)
    region = (125, 145, 59, 90)  # Adjust these coordinates
    
    last_image = None
    
    for idx, filename in enumerate(files):
        # Print progress every 1000 images
        if idx % 1000 == 0:
            print(f"Processing image {idx}: {filename} -- {datetime.now().time()}")
        
        current_path = os.path.join(screenshot_dir, filename)
        
        # Extract symbol and save cropped region
        current_symbol, img = extract_symbol(current_path, region, save_dir=cropped_dir, filename=filename)
        
        if img is not None:
            last_image = img.copy()
        
        if idx == 0:
            # Record the first finding
            changes.append({
                'frame': idx,
                'file': filename,
                'symbol': current_symbol,
                'type': 'initial'
            })
        elif previous is not None:
            symbol_changed = False
            
            # First check OCR results
            if current_symbol and previous['symbol']:
                symbol_changed = current_symbol != previous['symbol']
            else:
                # Fallback to image comparison
                previous_path = os.path.join(screenshot_dir, files[idx-1])
                symbol_changed = compare_regions(previous_path,
                                                 current_path,
                                                 region=region)
                
            if symbol_changed:
                changes.append({
                    'frame': idx,
                    'previous_file': files[idx-1],
                    'current_file': filename,
                    'from_symbol': previous['symbol'],
                    'to_symbol': current_symbol,
                    'type': 'change'
                })
        
        previous = {'symbol': current_symbol,
                    'path': current_path}
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(changes,
                  f,
                  indent=4)
    
    # Save last processed image with box
    if last_image is not None:
        save_image_with_box(last_image, region, output_path=output_image)
        
    print(f"Analysis complete. Found {len(changes)} entries (including initial). Results saved to {output_file}")
    print(f"Last processed image with box saved to {output_image}")
    print(f"Cropped regions saved to {cropped_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect stock symbol changes in TradingView screenshots')
    
    parser.add_argument('--screenshot-dir', required=True, help='Directory containing screenshots')
    parser.add_argument('--output', default='stock_changes.json', help='Output JSON file')
    parser.add_argument('--output-image', default='last_processed_with_box.jpg', help='Output image file with box')
    parser.add_argument('--cropped-dir', default='cropped_regions', help='Directory to save cropped regions of stock symbols')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.screenshot_dir):
        print(f"Error: Directory {args.screenshot_dir} does not exist!")
        exit(1)
        
    detect_changes(args.screenshot_dir, args.output, args.output_image, args.cropped_dir)
