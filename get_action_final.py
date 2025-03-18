import cv2
import numpy as np
from clean_hand_final import isolate_closest_object
#from black_seg_hand import init
import time
from skimage.metrics import hausdorff_distance  # For comparing contours TODO remove if not using
import black_seg_hand

# Global variables to store previous contour information
prev_contour_area = 0  # Already in use for saving images
prev_hand_contour = None  # New: stores the previous hand contour
prev_time = 0
DELAY_FRAMES = 100
MIN_HD_FOR_OK_GESTURE = 100
histogram = np.zeros(7)
action_to_drone = "Cannot classify gesture"

def get_action_to_drone(lock):
    global action_to_drone
    with lock:
        ret = action_to_drone        #lock to avoid conflicts
    return ret

def countdown_timer():
    for i in range(3, 0, -1):  # Countdown 3, 2, 1
        frame = np.zeros((500, 500, 3), dtype=np.uint8)  # Black screen
        cv2.putText(frame, str(i), (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 10)
        cv2.putText(frame, "PLACE HAND IN FRONT OF THE CAMERA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Countdown", frame)
        cv2.waitKey(1000)  # Wait for 1 second

    cv2.destroyAllWindows()  # Close the window after countdown


def speed_norm(frame, hand_contour):
    # Calculate the bounding rectangle
    x, y, w, h = cv2.boundingRect(hand_contour)
    # Calculate the aspect ratio of the bounding rectangle
    aspect_ratio = h / w  # Height-to-width ratio
    # Normalize the value between 0 and 1
    orientation_value = min(max((aspect_ratio - 1) / 3, 0.0), 1.0)  # Adjust scaling if needed
    # Draw the bounding rectangle and aspect ratio on the frame
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
    cv2.putText(frame, f"Orientation: {orientation_value:.2f}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame, orientation_value


def orientation_and_direction(frame, hand_contour):
    # Compute the convex hull and convexity defects
    hull = cv2.convexHull(hand_contour, returnPoints=False)
    defects = cv2.convexityDefects(hand_contour, hull)

    if defects is None:
        return frame, "Cannot classify gesture"

    # Calculate the moments to find the centroid of the hand
    M = cv2.moments(hand_contour)
    if M["m00"] == 0:
        return frame, "Cannot classify gesture"
    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

    # Draw the contour and centroid
    cv2.drawContours(frame, [hand_contour], -1, (0, 255, 0), 2)
    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    # Identify the farthest defect point (likely the pointing finger tip)
    max_distance = 0
    tip_point = None
    for i in range(defects.shape[0]):
        start_idx, end_idx, far_idx, _ = defects[i, 0]
        far = tuple(hand_contour[far_idx][0])
        distance = cv2.norm(np.array([cx, cy]) - np.array(far))
        if distance > max_distance:
            max_distance = distance
            tip_point = far

    if tip_point:
        cv2.circle(frame, tip_point, 8, (0, 0, 255), -1)
        # Classify the gesture based on the tip point position
        if abs(tip_point[0] - cx) > abs(tip_point[1] - cy):
            if tip_point[0] < cx:
                gesture = "left"
            else:
                gesture = "right"
        else:
            if tip_point[1] < cy:
                gesture = "up"
            else:
                gesture = "down"
        return frame, gesture
    return frame, "Cannot classify gesture"


def similar_bounding_boxes(contour1, contour2, iou_threshold=0.9):
    """
    Returns True if the Intersection over Union (IoU) between the bounding boxes of
    contour1 and contour2 is above the given threshold.
    """
    # Compute the bounding boxes for both contours
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    x2, y2, w2, h2 = cv2.boundingRect(contour2)

    # Calculate coordinates of the intersection rectangle
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    # Compute the area of intersection
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection = inter_width * inter_height

    # Compute the area of each bounding box
    area1 = w1 * h1
    area2 = w2 * h2

    # Compute the area of the union
    union = area1 + area2 - intersection
    iou = intersection / union if union > 0 else 0

    # For debugging, you can print the IoU value
    # print("IoU:", iou)

    return iou >= iou_threshold


def detect_gesture(frame, mask):
    global prev_hand_contour  # To store the previous contour for comparison

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return frame, "Cannot classify gesture", 0.0
    # Use the largest contour as the hand
    hand_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(hand_contour)
    x, y, w, h = cv2.boundingRect(hand_contour)
    rect_area = w * h
    ratio = contour_area / rect_area if rect_area > 0 else 0.0

    # Get speed/orientation value
    frame, velocity = speed_norm(frame, hand_contour)
    # Get gesture direction (up/down/left/right)
    frame, gesture = orientation_and_direction(frame, hand_contour)

    # Update the previous hand contour (make a copy to avoid unwanted modifications)

    # Decide on the final action
    if save_image(hand_contour, mask):  # capture image
        action = "picture"
    elif ratio > 0.55:
        action = "forward"
    else:
        action = gesture
        velocity = 0

    prev_hand_contour = hand_contour.copy()

    return frame, action, velocity


def save_image(hand_contour, masked_frame):
    # Get bounding box for the hand
    x, y, w, h = cv2.boundingRect(hand_contour)
    box_ration = h / w
    
    # Define the TOP 10% region
    top_box_height = max(1, h // 10)
    top_y = y
    top_x = x
    top_w = w
    top_h = top_box_height

    # --- IMPORTANT: Extract the region *before* drawing on 'masked_frame' ---
    top_region = masked_frame[top_y:top_y + top_h, top_x:top_x + top_w]

    # Apply thresholding (helps keep contours distinct)
    # Adjust threshold value if your background is not pure black or white
    _, top_region_bin = cv2.threshold(top_region, 127, 255, cv2.THRESH_BINARY)

    # (Optional) Morphological open can separate close blobs
    kernel = np.ones((3, 3), np.uint8)
    top_region_bin = cv2.morphologyEx(top_region_bin, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours in the thresholded top region using RETR_TREE
    top_contours, _ = cv2.findContours(top_region_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



    return (len(top_contours) == 2) and (box_ration > 1.5)


def detect_gesture_wrapper(frame, mask):
    gesture_value = {"up": 0, "down": 1, "right": 2, "left": 3, "forward": 4, "picture": 5, "Cannot classify gesture": 6}
    global histogram
    global action_to_drone

    processed_frame, gesture, orientation_value = detect_gesture(frame, mask)
    histogram[gesture_value[gesture]] += 1  # Default to "Cannot classify gesture" if missing

    if histogram.sum() > DELAY_FRAMES:
        max_index = np.argmax(histogram)
        if histogram[max_index] > DELAY_FRAMES * 0.8:           #take the majority action only if we are 80% certain it is the action intended
            ret_gesture = [k for k, v in gesture_value.items() if v == max_index]
            action_to_drone = ret_gesture[0]
        else:
            action_to_drone = "Cannot classify gesture"
        # if histogram[gesture_value["picture"]] > 1:
        #     action_to_drone = "picture"
        #     processed_frame = np.full(processed_frame.shape, 255, dtype=np.uint8)       # Flash screen when taking an image
        print(action_to_drone)
        histogram = np.zeros(7)

    return processed_frame, gesture, orientation_value

def get_action(frame, mask):

    #frame = cv2.GaussianBlur(frame, (5, 5), 0)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # Find contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return np.zeros_like(frame)  # Return an empty frame if there aren't at least two contours

    # Sort contours by area in descending order
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Get the two largest contours
    largest_contours = sorted_contours[:2]

    # Find the rightmost contour
    def get_rightmost_x(contour):
        return max(point[0][0] for point in contour)

    rightmost_contour = max(largest_contours, key=get_rightmost_x)

    # Create a new mask for the rightmost contour
    rightmost_contour_mask = np.zeros_like(mask)
    cv2.drawContours(rightmost_contour_mask, [rightmost_contour], -1, 255, thickness=cv2.FILLED)

    # Apply the mask to the original frame
    frame = cv2.bitwise_and(frame, frame, mask=rightmost_contour_mask)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Detect gesture (which now may return "ok" if a significant contour change is detected)
    processed_frame, gesture, orientation_value = detect_gesture_wrapper(frame, mask)

    return gesture
    # Display the gesture and orientation value on the frame
    # cv2.putText(processed_frame, gesture, (50, 50),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)





# --- Main Video Loop ---
def init_get_action():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    pixels = []
    count = 0
    model = None
    prev_box = None  # Will be stored as full-frame (x, y, w, h)
    kalman = None
    countdown_timer()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame = cv2.flip(frame, 1)  # Remove mirror effect
        original_frame = frame
        original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)

        # Initialize prev_box in full-frame coordinates on the first frame
        if prev_box is None:
            roi, roi_coords = black_seg_hand.get_square_roi(frame)
            start_x, start_y, end_x, end_y = roi_coords
            prev_box = (start_x, start_y, end_x - start_x, end_y - start_y)
            kalman = black_seg_hand.initialize_kalman(prev_box)

        (count, pixels, model, segmented_mask,
         largest_contour, roi_coords, roi) = black_seg_hand.process_frame(frame, count, pixels, model, prev_box)

        segmented_color = black_seg_hand.color_map(roi, segmented_mask)
        frame, mask = black_seg_hand.update_display(frame, roi_coords, segmented_color, largest_contour)
        #original_frame = original_frame * mask

        if largest_contour is not None:
            x, y, w, h = cv2.boundingRect(largest_contour)
            # Convert the bounding box from ROI-relative to full-frame coordinates
            current_box = (roi_coords[0] + x, roi_coords[1] + y, w, h)
            tracked_box, direction_vector = black_seg_hand.tracking(prev_box, current_box, kalman=kalman)
            prev_box = tracked_box  # Update for the next frame

            if tracked_box is not None:
                x_box, y_box, w_box, h_box = tracked_box
                cv2.rectangle(frame, (x_box, y_box), (x_box + w_box, y_box + h_box), (0, 0, 255), 2)
                cv2.putText(frame, f"dx: {direction_vector[0]:.1f}, dy: {direction_vector[1]:.1f}",
                            (x_box, y_box - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                action = get_action(original_frame, mask)
                cv2.putText(frame, str(action), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Skin Detection with Gaussian Model and Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    init_get_action()