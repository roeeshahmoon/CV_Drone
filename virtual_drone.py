import pygame
import math
from get_action_final import init_get_action
from get_action_final import get_action_to_drone
import threading
import time

lock = threading.Lock()

# Initialize Pygame
def game():
    
    pygame.init()

    # Screen settings
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Drone Simulator")

    # Colors
    WHITE = (255, 255, 255)
    BLUE = (0, 100, 255)

    # Drone settings
    drone_size = (80, 40)  # Width, Height
    drone_x, drone_y = WIDTH // 2, HEIGHT // 2  # Start in the center
    drone_angle = 0  # Facing right (0 degrees)

    # Movement step sizes
    MOVE_STEP = 10
    ROTATION_STEP = 90

    # Load drone image (or draw manually)
    drone_image = pygame.Surface(drone_size, pygame.SRCALPHA)
    pygame.draw.polygon(drone_image, BLUE, [(0, 20), (60, 0), (80, 20), (60, 40), (0, 20)])  # Simple drone shape

    # Main loop
    running = True
    while running:
        screen.fill(WHITE)  # Clear screen

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        time.sleep(2)
        action = get_action_to_drone(lock)

        # Calculate new position before applying movement
        new_x = drone_x
        new_y = drone_y

        if action == "forward":  # Move Forward
            new_x += MOVE_STEP * math.cos(math.radians(drone_angle))
            new_y -= MOVE_STEP * math.sin(math.radians(drone_angle))

        if action == "left":  # Rotate Left (Counter-clockwise)
            drone_angle += ROTATION_STEP

        if action == "right":  # Rotate Right (Clockwise)
            drone_angle -= ROTATION_STEP

        # **New Actions for Up/Down**
        if action == "up":  # Increase drone size
            drone_size = (drone_size[0] + 5, drone_size[1] + 3)  # Increase width & height
            drone_size = min(drone_size[0], 200), min(drone_size[1], 120)  # Limit max size

        if action == "down":  # Decrease drone size
            drone_size = (drone_size[0] - 5, drone_size[1] - 3)  # Decrease width & height
            drone_size = max(drone_size[0], 40), max(drone_size[1], 20)  # Limit min size

        # Prevent angle overflow
        drone_angle %= 360

        # **Recreate the drone image with the updated size**
        drone_image = pygame.Surface(drone_size, pygame.SRCALPHA)
        pygame.draw.polygon(drone_image, BLUE, [(0, drone_size[1] // 2), 
                                                (drone_size[0] * 0.75, 0), 
                                                (drone_size[0], drone_size[1] // 2), 
                                                (drone_size[0] * 0.75, drone_size[1]), 
                                                (0, drone_size[1] // 2)])  # Updated drone shape

        # Get the rotated drone rectangle to check boundaries
        rotated_drone = pygame.transform.rotate(drone_image, -drone_angle)
        drone_rect = rotated_drone.get_rect(center=(new_x, new_y))

        # **Boundary Check:** Keep drone inside the screen
        if 0 <= drone_rect.left and drone_rect.right <= WIDTH and 0 <= drone_rect.top and drone_rect.bottom <= HEIGHT:
            drone_x, drone_y = new_x, new_y  # Update position **only if within boundaries**

        # Draw drone
        screen.blit(rotated_drone, drone_rect.topleft)

        # Update screen
        pygame.display.flip()
        pygame.time.delay(50)  # Small delay to control speed

    pygame.quit()



if __name__ == "__main__":
    thread1 = threading.Thread(target=init_get_action)
    thread2 = threading.Thread(target=game)

    # Start threads
    thread1.start()
    thread2.start()

    # Wait for both threads to complete
    thread1.join()
    thread2.join()



























# from djitellopy import Tello
# import cv2
# import os
# import time
# from get_action_final import init_get_action
# from get_action_final import get_action_to_drone
# import threading

# lock = threading.Lock()



# def fly_drone():
#     drone = Tello()
#     drone.connect()

#     try:
#         picture_id = 1  # To keep track of picture filenames
#         drone.streamon()
#         time.sleep(2)

#         while True:
#             response = drone.send_command_with_return("command")
#             if response is None or response.lower() != "ok":
#                 print("Connection failed! No response from drone.")
#                 break

#             display_drone_video(drone)
#             action = get_action_to_drone(lock)

#             # Exit the loop when 'q' is pressed
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#             if action == "up" and drone.get_height() < 10:
#                 drone.takeoff()

#             elif action == "down" and drone.get_height() > 10:
#                 drone.land()
#                 break  # Exit the loop after landing
            
#             elif action == "down":
#                 drone.move_down(20)

#             elif action == "up":
#                 drone.move_up(20)

#             elif action == "forward":
#                 #if velocity > 0.1:
#                 drone.move_forward(40)

#             elif action == "left":
#                 drone.rotate_counter_clockwise(45)  # Rotate left by 45 degrees

#             elif action == "right":
#                 drone.rotate_clockwise(45)  # Rotate right by 45 degrees

#             elif action == "picture":
#                 take_picture(drone, picture_id)
#                 picture_id += 1

#             elif action == "Cannot classify gesture":
#                 time.sleep(2)
            
#             elif cv2.waitKey(1) & 0xFF == ord('l'):
#                 print("Emergency landing triggered by 'l' key.")
#                 drone.land()
#                 break

#             else:
#                 print(f"Unknown action: {action}")


#     except Exception as e:
#         print("An error occurred:", e)

#     finally:
#         print("Landing and disconnecting...")
#         drone.land()
#         drone.streamoff()
#         drone.end()
#         time.sleep(2)
#         cv2.destroyAllWindows()


# if __name__ == "__main__":
#     thread1 = threading.Thread(target=init_get_action)
#     thread2 = threading.Thread(target=fly_drone)

#     # Start threads
#     thread1.start()
#     thread2.start()

#     # Wait for both threads to complete
#     thread1.join()
#     thread2.join()