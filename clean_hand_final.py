import cv2
import numpy as np
def adjust_hsv_range(frame):
    def nothing(x):
        pass

    # Create trackbars for HSV range
    cv2.namedWindow("Adjust HSV")
    cv2.createTrackbar("H Lower", "Adjust HSV", 0, 179, nothing)
    cv2.createTrackbar("H Upper", "Adjust HSV", 20, 179, nothing)
    cv2.createTrackbar("S Lower", "Adjust HSV", 30, 255, nothing)
    cv2.createTrackbar("S Upper", "Adjust HSV", 150, 255, nothing)
    cv2.createTrackbar("V Lower", "Adjust HSV", 60, 255, nothing)
    cv2.createTrackbar("V Upper", "Adjust HSV", 255, 255, nothing)

    while True:
        h_lower = cv2.getTrackbarPos("H Lower", "Adjust HSV")
        h_upper = cv2.getTrackbarPos("H Upper", "Adjust HSV")
        s_lower = cv2.getTrackbarPos("S Lower", "Adjust HSV")
        s_upper = cv2.getTrackbarPos("S Upper", "Adjust HSV")
        v_lower = cv2.getTrackbarPos("V Lower", "Adjust HSV")
        v_upper = cv2.getTrackbarPos("V Upper", "Adjust HSV")

        hsv_lower = (h_lower, s_lower, v_lower)
        hsv_upper = (h_upper, s_upper, v_upper)

        mask = cv2.inRange(frame, hsv_lower, hsv_upper)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return hsv_lower, hsv_upper

def isolate_closest_object(frame, hsv_lower, hsv_upper):
    """
    Isolate the rightmost object among the top two largest contours in the HSV range.

    Args:
        frame (numpy.ndarray): The input frame from the camera.
        hsv_lower (tuple): Lower HSV threshold for object detection.
        hsv_upper (tuple): Upper HSV threshold for object detection.

    Returns:
        numpy.ndarray: The processed frame with only the selected object visible.
    """
    # Apply Gaussian blur to reduce noise
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the specified HSV range
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    # Find contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) < 2:
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
    result = cv2.bitwise_and(frame, frame, mask=rightmost_contour_mask)

    return result
