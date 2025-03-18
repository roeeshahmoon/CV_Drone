import cv2
import numpy as np

# Constants and thresholds
SQUARE_SIZE = 400  # Adjust this to fit your hand comfortably
MAX_FRAMES = 100
IGNORE_START_FRAMES = 10
KALMAN_WARMUP_FRAMES = 100
FULL_FRAME_SEGMENTATION_START = IGNORE_START_FRAMES + MAX_FRAMES + KALMAN_WARMUP_FRAMES
lower_skin = np.array([0, 80, 60], dtype=np.uint8)
upper_skin = np.array([20, 150, 255], dtype=np.uint8)
THRESHOLD = 40  # Mahalanobis distance threshold

# --------------------------------------------------
# Gaussian Model Functions
# --------------------------------------------------
def fit(pixels):
    pixels = np.asarray(pixels, dtype=np.float64)
    mean = np.mean(pixels, axis=0)
    cov = np.cov(pixels, rowvar=False)
    inv_cov = np.linalg.inv(cov + 1e-8 * np.eye(3))
    return {"mean": mean, "inv_cov": inv_cov, "threshold": THRESHOLD}

def are_pixels_in_distribution(model, image):
    mean = model["mean"]
    inv_cov = model["inv_cov"]
    threshold = model["threshold"]

    reshaped = image.reshape(-1, 3).astype(np.float64)
    diff = reshaped - mean
    md_sq = np.sum(diff @ inv_cov * diff, axis=1)
    mask = md_sq < threshold
    mask = mask.reshape(image.shape[:2])
    return mask.astype(np.uint8)

# --------------------------------------------------
# Segmentation and Post-Processing Functions
# --------------------------------------------------
def segment_with_threshold(frame):
    mask = cv2.inRange(frame, lower_skin, upper_skin)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    return (mask == 255).astype(int)

def post_process_and_find_hand(mask):
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if np.max(mask) == 1:
        mask = (mask * 255).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

def color_map(frame, segmented_mask):
    # Change colors: background becomes black [0,0,0] and hand becomes white [255,255,255]
    colors = [[0, 0, 0], [255, 255, 255]]
    segmented_color = np.zeros(frame.shape, dtype=np.uint8)
    for i in range(len(colors)):
        segmented_color[segmented_mask == i] = colors[i]
    return segmented_color


# --------------------------------------------------
# ROI and Visualization Helpers
# --------------------------------------------------
def get_square_roi(frame):
    height, width, _ = frame.shape
    start_x = (width - SQUARE_SIZE) // 2
    start_y = (height - SQUARE_SIZE) // 2
    end_x = start_x + SQUARE_SIZE
    end_y = start_y + SQUARE_SIZE
    roi = frame[start_y:end_y, start_x:end_x]
    return roi, (start_x, start_y, end_x, end_y)

def update_display(frame, roi_coords, segmented_color, largest_contour):
    start_x, start_y, end_x, end_y = roi_coords
    if not (start_x == 0 and start_y == 0 and end_x == frame.shape[1] and end_y == frame.shape[0]):
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    if largest_contour is not None:
        cv2.drawContours(segmented_color, [largest_contour], -1, (255, 0, 0), 2)

    frame[start_y:end_y, start_x:end_x] = segmented_color
    
    # Create a black frame with only the ROI visible
    masked_frame = np.zeros_like(frame)
    masked_frame[start_y:end_y, start_x:end_x] = segmented_color

    return frame, masked_frame

# --------------------------------------------------
# Tracking Function (Updated)
# --------------------------------------------------
def tracking(bounding_box1, bounding_box2, kalman=None):
    # Both bounding boxes are assumed to be in full-frame coordinates in the format (x, y, w, h)
    if bounding_box2 is not None and bounding_box1 is None:
        return bounding_box2, [0, 0]
    if bounding_box1 is None or bounding_box2 is None:
        return None, None

    x1, y1, w1, h1 = bounding_box1
    x2, y2, w2, h2 = bounding_box2

    dx = (x2 + w2 / 2) - (x1 + w1 / 2)
    dy = (y2 + h2 / 2) - (y1 + h1 / 2)
    direction_vector = (dx, dy)

    if kalman is not None:
        measurement = np.array([[x2], [y2], [w2], [h2]], dtype=np.float32)
        if kalman.statePre is None or kalman.statePost.shape != (4, 1):
            kalman.statePost = measurement.copy()
        kalman.correct(measurement)
        predicted = kalman.predict()
        x_k = int(predicted[0, 0])
        y_k = int(predicted[1, 0])
        w_k = int(predicted[2, 0])
        h_k = int(predicted[3, 0])

        add = 100
        bounding_box_2 = (max(x_k - add, 0), max(y_k - add, 0), int(w_k + add * 2), int(h_k + add * 2))
    else:
        bounding_box_2 = (x2, y2, w2, h2)

    return bounding_box_2, direction_vector

# --------------------------------------------------
# Main Processing Loop
# --------------------------------------------------
def process_frame(frame, count, pixels, model, prev_box):
    height, width, _ = frame.shape
    use_full_frame = model is not None and count >= FULL_FRAME_SEGMENTATION_START

    if use_full_frame:
        # Use the previously tracked full-frame bounding box as ROI
        x, y, w, h = prev_box
        roi = frame[y:y+h, x:x+w]
        roi_coords = (x, y, x+w, y+h)
    else:
        roi, roi_coords = get_square_roi(frame)

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    segmented_mask = segment_with_threshold(hsv_roi)
    largest_contour = None

    if count < IGNORE_START_FRAMES:
        pass
    elif count < IGNORE_START_FRAMES + MAX_FRAMES and model is None:
        mask = (segmented_mask == 1)
        pixels.extend(hsv_roi[mask])
    elif model is None:
        model = fit(pixels)
    else:
        skin_mask = are_pixels_in_distribution(model, hsv_roi)
        segmented_mask = skin_mask.astype(int)
        largest_contour = post_process_and_find_hand(segmented_mask)

    count += 1
    return count, pixels, model, segmented_mask, largest_contour, roi_coords, roi

def initialize_kalman(initial_box):
    kalman = cv2.KalmanFilter(4, 4)
    kalman.transitionMatrix = np.eye(4, dtype=np.float32)
    kalman.measurementMatrix = np.eye(4, dtype=np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
    kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
    kalman.errorCovPost = np.eye(4, dtype=np.float32) * 0.1
    kalman.statePost = np.zeros((4, 1), dtype=np.float32)
    if initial_box:
        x, y, w, h = initial_box
        kalman.statePost = np.array([[x], [y], [w], [h]], dtype=np.float32)
    return kalman

# --------------------------------------------------
# Main Function (Updated)
# --------------------------------------------------
def init():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    pixels = []
    count = 0
    model = None
    prev_box = None  # Will be stored as full-frame (x, y, w, h)
    kalman = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame = cv2.flip(frame, 1)  # Remove mirror effect

        # Initialize prev_box in full-frame coordinates on the first frame
        if prev_box is None:
            roi, roi_coords = get_square_roi(frame)
            start_x, start_y, end_x, end_y = roi_coords
            prev_box = (start_x, start_y, end_x - start_x, end_y - start_y)
            kalman = initialize_kalman(prev_box)

        (count, pixels, model, segmented_mask,
         largest_contour, roi_coords, roi) = process_frame(frame, count, pixels, model, prev_box)

        segmented_color = color_map(roi, segmented_mask)
        frame = update_display(frame, roi_coords, segmented_color, largest_contour)

        if largest_contour is not None:
            x, y, w, h = cv2.boundingRect(largest_contour)
            # Convert the bounding box from ROI-relative to full-frame coordinates
            current_box = (roi_coords[0] + x, roi_coords[1] + y, w, h)
            tracked_box, direction_vector = tracking(prev_box, current_box, kalman=kalman)
            prev_box = tracked_box  # Update for the next frame

            if tracked_box is not None:
                x_box, y_box, w_box, h_box = tracked_box
                cv2.rectangle(frame, (x_box, y_box), (x_box + w_box, y_box + h_box), (0, 0, 255), 2)
                cv2.putText(frame, f"dx: {direction_vector[0]:.1f}, dy: {direction_vector[1]:.1f}",
                            (x_box, y_box - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow('Skin Detection with Gaussian Model and Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    init()
