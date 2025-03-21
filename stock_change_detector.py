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

def extract_symbol(img_path, region=(88, 103, 53, 92)):
    """Extract stock symbol using OCR from specified region."""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None, None, None
            
        # Crop to symbol region (top-Y, bottom-Y, left-X, right-X)
        y1, y2, x1, x2 = region
        symbol_region = img[y1:y2, x1:x2]
        
        # Enhanced preprocessing pipeline
        gray = cv2.cvtColor(symbol_region, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        scaled = cv2.resize(inverted, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        thresh = cv2.adaptiveThreshold(scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        denoised = cv2.medianBlur(thresh, 3)
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.dilate(denoised, kernel, iterations=1)

        custom_config = r'--oem 2 --psm 7 --dpi 300 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        symbol = pytesseract.image_to_string(processed, lang="eng", config=custom_config)
        
        return symbol.strip(), img, symbol_region
        
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return None, None, None

def compare_regions(img1, img2, threshold=0.9):
    """Compare similarity of regions using SSIM."""
    try:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        score, _ = structural_similarity(gray1, gray2, full=True)
        return score < threshold
    except Exception as e:
        print(f"Comparison error: {str(e)}")
        return True

def save_image_with_box(img, region, output_path):
    """Save image with a box around the specified region."""
    y1, y2, x1, x2 = region
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(output_path, img)

def detect_changes(screenshot_dir, output_file, output_image, cropped_dir):
    """Main detection logic with optimized processing."""
    files = sorted([f for f in os.listdir(screenshot_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    changes = []
    previous = None
    region = (88, 103, 53, 92)
    last_image = None
    
    # Create output directory for cropped regions
    os.makedirs(cropped_dir, exist_ok=True)

    for idx, filename in enumerate(files):
        if idx % 1000 == 0:
            print(f"Processing image {idx}: {filename} -- {datetime.now().time()}")

        current_path = os.path.join(screenshot_dir, filename)
        current_symbol, img, symbol_region = extract_symbol(current_path, region)
        
        if img is not None:
            last_image = img.copy()

        if idx == 0:
            # Save first frame's cropped region
            cv2.imwrite(os.path.join(cropped_dir, filename), symbol_region)
            changes.append({
                'frame': idx,
                'file': filename,
                'symbol': current_symbol,
                'type': 'initial'
            })
            previous = {
                'symbol': current_symbol,
                'symbol_region': symbol_region,
                'path': current_path
            }
            continue

        symbol_changed = False
        if current_symbol and previous['symbol']:
            symbol_changed = current_symbol != previous['symbol']
        else:
            # Compare regions directly in memory
            symbol_changed = compare_regions(previous['symbol_region'], symbol_region)

        if symbol_changed:
            # Save current and previous cropped regions
            cv2.imwrite(os.path.join(cropped_dir, filename), symbol_region)
            cv2.imwrite(os.path.join(cropped_dir, f"prev_{filename}"), previous['symbol_region'])
            
            changes.append({
                'frame': idx,
                'previous_file': f"prev_{filename}",
                'current_file': filename,
                'from_symbol': previous['symbol'],
                'to_symbol': current_symbol,
                'type': 'change'
            })
            previous = {
                'symbol': current_symbol,
                'symbol_region': symbol_region,
                'path': current_path
            }

    # Save results
    with open(output_file, 'w') as f:
        json.dump(changes, f, indent=4)

    if last_image is not None:
        save_image_with_box(last_image, region, output_path=output_image)
        
    print(f"Analysis complete. Found {len(changes)} changes. Results saved to {output_file}")
    print(f"Last processed image with box saved to {output_image}")
    print(f"Changed cropped regions saved to {cropped_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimized stock symbol change detection')
    parser.add_argument('--screenshot-dir', required=True, help='Directory containing screenshots')
    parser.add_argument('--output', default='stock_changes.json', help='Output JSON file')
    parser.add_argument('--output-image', default='last_processed_with_box.jpg', help='Output image file with box')
    parser.add_argument('--cropped-dir', default='changed_cropped', help='Directory for changed cropped regions')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.screenshot_dir):
        print(f"Error: Directory {args.screenshot_dir} does not exist!")
        exit(1)
        
    detect_changes(args.screenshot_dir, args.output, args.output_image, args.cropped_dir)
