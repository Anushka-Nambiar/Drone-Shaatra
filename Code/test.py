from Drone import Drone
import getch

drone = Drone()
drone.calibrate()
while 1:
    drone.control(str(getch.getch()))
