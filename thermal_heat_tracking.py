import cv2
import numpy as np

# === CONFIGURATION ===
THERMAL_CAMERA_INDEX = 2  # Change if your HIKMICRO camera uses a different index
WINDOW_NAME = "Thermal Heat Tracking"


def find_hottest_point(frame):
    """
    Given a thermal image frame, find the hot region (area > 200 pixels) with the greatest total heat (sum of pixel values).
    Returns (x, y, max_value) for the centroid of the hottest region.
    """
    # If the frame is colored, convert to grayscale (thermal cameras may output grayscale or color)
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # Threshold to find hot regions (use a high threshold, e.g., 90% of max)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(gray)
    thresh_val = maxVal * 0.9
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    # Find contours of hot regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area > 200 pixels
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 200]
    if not valid_contours:
        # Fallback to hottest pixel if no valid region
        x, y = maxLoc
        return x, y, maxVal

    # Find the contour with the greatest total heat
    max_heat = -1
    best_cx, best_cy, best_max = 0, 0, 0
    for cnt in valid_contours:
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        region_vals = gray[mask == 255]
        total_heat = np.sum(region_vals)
        if total_heat > max_heat:
            max_heat = total_heat
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0
            region_max = float(np.max(region_vals)) if region_vals.size > 0 else maxVal
            best_cx, best_cy, best_max = cx, cy, region_max

    return best_cx, best_cy, best_max


def main():
    print("Starting HIKMICRO Thermal Heat Tracking...")
    cap = cv2.VideoCapture(THERMAL_CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open thermal camera at index {THERMAL_CAMERA_INDEX}")
        return

    print("Press ESC to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from thermal camera.")
            break

        # Flip the frame by 180 degrees
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Crop out top 40 pixels and right 30 pixels
        frame = frame[70:, :-80]

        # Find the hottest point
        x, y, maxVal = find_hottest_point(frame)

        # Draw a circle at the hottest point
        display = frame.copy()
        cv2.circle(display, (x, y), 8, (0, 0, 255), 2)
        cv2.putText(display, f"Hot: ({x}, {y}) {maxVal:.1f}", (x+10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show the frame
        cv2.imshow(WINDOW_NAME, display)

        # Print the hottest point info
        print(f"Hottest point: x={x}, y={y}, value={maxVal:.1f}", end='\r')

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nTracking stopped.")


if __name__ == "__main__":
    main() 