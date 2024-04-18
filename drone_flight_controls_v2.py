"""Module for controlling drone flight operations.

This module provides functions for connecting to a drone, arming/disarming,
taking off, moving in specified directions, landing, and disarming after landing.
"""
import time
from pymavlink import mavutil


# Create the connection
def connect_to_drone(connection_str):
    """
    Connect to the drone using the specified connection string.

    Args:
        connection_str (str): The connection string used to connect to the drone.

    Returns:
        mavutil.mavlink_connection: A connection object if successful, None otherwise.
    """
    # telemetry baud = 57600 usb baud = 115200
    # Connect to the drone
    connection = mavutil.mavlink_connection(connection_str, baud=57600)
    # wait for the heartbeat message
    msg = connection.recv_match(type='HEARTBEAT', blocking=True, timeout=10)

    if msg:
        print(f"Heartbeat received from system (ID: {msg.get_srcSystem()})")
        return connection

    print("Connection failed: No heartbeat received.")
    return None

# Set the mode of the drone
def set_mode_guided(connection):
    """
    Set the drone's mode to GUIDED.

    Args:
        connection (mavutil.mavlink_connection): The connection to the drone.

    Returns:
        bool: True if the mode was successfully set, False otherwise.
    """
    connection.mav.command_long_send(
        connection.target_system, connection.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, # Base mode
        mavutil.mavlink.COPTER_MODE_GUIDED, 0, 0, 0, 0, 0 # Custom mode - GUIDED for Copter
    )

    # Wait for the COMMAND_ACK message to be received
    msg = connection.recv_match(type='COMMAND_ACK', blocking=True)
    if not msg:
        print("No COMMAND_ACK message received for mode set")
        return False

    if msg.command == mavutil.mavlink.MAV_CMD_DO_SET_MODE and msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
        print("GUIDED mode set successfully")
        return True

    print(f"Failed to set GUIDED mode: {msg.result}")
    return False



# Arm the drone, pass the connection and command. command can be 0 or 1 0: Disarm, 1: Arm
def arm_disarm_drone(connection, command):
    """
    Send a command to arm or disarm the drone.

    Args:
        connection (mavutil.mavlink_connection): The connection to the drone.
        command (int): 1 to arm the drone, 0 to disarm it.

    Returns:
        bool: True if the command was successful, False otherwise.
    """

    connection.mav.command_long_send(
        connection.target_system, connection.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,0, command, 21196, 0, 0, 0, 0, 0)

    # Wait for the COMMAND_ACK message to be received
    msg = connection.recv_match(type='COMMAND_ACK', blocking=True)
    if not msg:
        print("No COMMAND_ACK message received")
    # Check if the command was successful
    if msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
        if msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            print("Arm/Disarm command acknowledged and executed")
            return True

        print(f"Arm/Disarm command failed with result: {msg.result}")
    else:
        print("Received COMMAND_ACK message with unexpected command type")
    return False

# hover the drone specifying altitude
def takeoff_drone(connection, altitude):
    """
    Instruct the drone to take off and hover at a specified altitude.

    Args:
        connection (mavutil.mavlink_connection): The connection to the drone.
        altitude (float): The altitude in meters that the drone should hover at.

    Returns:
        bool: True if the takeoff command was successful, False otherwise.
    """

    connection.mav.command_long_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, altitude)

    msg_takeoff = connection.recv_match(type='COMMAND_ACK', blocking=True)
    if not msg_takeoff or msg_takeoff.result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
        print(f"Failed to takeoff: {msg_takeoff.result}")
        return False
    print(f"Drone took off successfully with altitude: {altitude}m")
    return True

# move forward x meters
def move_forward(connection, x_distance):
    """
    Instruct the drone to take move forward a certian distance.

    Args:
        connection (mavutil.mavlink_connection): The connection to the drone.
        x_distance: The distance in meters that the drone should move forward for.

    Returns:
        Boolean: True if the drone moved forward successfully, False otherwise.
    """
    try:
        velocity = 1
        move_time = x_distance / velocity
        type_mask = int(0b0000111111000111)
        # Command to move forward
        connection.mav.set_position_target_local_ned_send(
            0, connection.target_system,
            connection.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,type_mask, 0, 0, 0, velocity, 0, 0, 0, 0, 0, 0, 0 )


        # Wait for the drone to move the desired distance
        time.sleep(move_time)

        connection.mav.set_position_target_local_ned_send(
            0, connection.target_system,
            connection.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,type_mask, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        return True

    except Exception as e:
        print(f"Error: Failed to move the drone forward due to {e}.")
        return False


# move backward x meters
def move_backward(connection, x_distance):
    """
    Instruct the drone to take move backward a certian distance.

    Args:
        connection (mavutil.mavlink_connection): The connection to the drone.
        x_distance: The distance in meters that the drone should move backward for.

    Returns:
        Boolean: True if the drone moved backward successfully, False otherwise.
    """
    try:
        velocity = -1
        move_time = x_distance / abs(velocity)
        type_mask = int(0b0000111111000111)
        # Command to move backward
        connection.mav.set_position_target_local_ned_send(
            0, connection.target_system,
            connection.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,type_mask, 0, 0, 0, velocity, 0, 0, 0, 0, 0, 0, 0 )

        # Wait for the drone to move the desired distance
        time.sleep(move_time)

        connection.mav.set_position_target_local_ned_send(
            0, connection.target_system,
            connection.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,type_mask, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )

        return True


    except Exception as e:
        print(f"Error: Failed to move the drone forward due to {e}.")
        return False

# move right y meters
def move_right(connection, y_distance):
    """
    Instruct the drone to take move right a certian distance.

    Args:
        connection (mavutil.mavlink_connection): The connection to the drone.
        y_distance: The distance in meters that the drone should move right for.

    Returns:
        Boolean: True if the drone moved right successfully, False otherwise.
    """
    try:
        velocity = 1
        move_time = y_distance / velocity
        type_mask = int(0b0000111111000111)
        # Command to move right
        connection.mav.set_position_target_local_ned_send(
            0, connection.target_system,
            connection.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,type_mask, 0, 0, 0, 0, velocity, 0, 0, 0, 0, 0, 0 )

        # Wait for the drone to move the desired distance
        time.sleep(move_time)

        connection.mav.set_position_target_local_ned_send(
            0, connection.target_system,
            connection.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,type_mask, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )


        return True

    except Exception as e:
        print(f"Error: Failed to move the drone forward due to {e}.")
        return False

def move_left(connection, y_distance):
    """
    Instruct the drone to take move left a certian distance.

    Args:
        connection (mavutil.mavlink_connection): The connection to the drone.
        y_distance: The distance in meters that the drone should move left for.

    Returns:
        Boolean: True if the drone moved left successfully, False otherwise.
    """
    try:
        velocity = -1
        move_time = y_distance / abs(velocity)
        type_mask = int(0b0000111111000111)
        # Command to move left
        connection.mav.set_position_target_local_ned_send(
            0, connection.target_system,
            connection.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,type_mask, 0, 0, 0, 0, velocity, 0, 0, 0, 0, 0, 0 )

        # Wait for the drone to move the desired distance
        time.sleep(move_time)

        connection.mav.set_position_target_local_ned_send(
            0, connection.target_system,
            connection.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,type_mask, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )


        return True

    except Exception as e:
        print(f"Error: Failed to move the drone forward due to {e}.")
        return False

def land_drone(connection):
    """
    Instructs the drone to land at its current location.

    Sends a MAV_CMD_NAV_LAND command to the drone via the specified connection,
    requesting the drone to initiate its landing sequence. The function then waits
    for an acknowledgment (COMMAND_ACK) message from the drone, checking if the
    landing command was accepted and initiated successfully.

    Args:
        connection (mavutil.mavlink_connection): The connection object representing
            the MAVLink communication link to the drone.

    Returns:
        bool: True if the drone accepted the landing command and is attempting to land,
              False if the landing command failed or was rejected.
    """
    connection.mav.command_long_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0, 0, 0, 0, 0, 0, 0, 0)

    msg_land = connection.recv_match(type='COMMAND_ACK', blocking=True)
    if not msg_land or msg_land.result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
        print(f"Failed to land: {msg_land}")
        return False

    print("Drone landed successfully")
    return True

def return_home(connection):
    """
    instructs the drone to retun to home location.

    Args:
        connection (mavutil.mavlink_connection): The connection to the drone.

    Returns:
        Boolean: True if the drone returned home successfully, False otherwise.
    """

    try:
        connection.mav.command_long_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
        0, 0, 0, 0, 0, 0, 0, 0)

        print("Command was accepted and executed successfully.")
        return True

    except Exception as e:
        print(f"Error: Failed to return home due to {e}.")
        print("Command was not executed successfully. Result code:", connection.result)
        return False



def rotate_drone(connection, angle, clockwise):

    """
    Instruct the drone to rotate a certain angle.

    Args:
        connection (mavutil.mavlink_connection): The connection to the drone.
        angle: The angle in degrees that the drone should rotate.
        clockwise: The direction of rotation, 1 for clockwise, -1 for counter-clockwise.

    Returns:
        Boolean: True if the drone rotated successfully, False otherwise.
    """

    t = angle / 25
    try:
        connection.mav.command_long_send(
            connection.target_system,
            connection.target_component,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            0, angle, 25, clockwise, 1, 0, 0, 0, 0)

        time.sleep(t)

        return True

    except Exception as e:
        print(f"Error: Failed to rotate the drone due to {e}.")
        return False




def main():
    """Main function for the module. Demonstrates the usage of the functions in the module."""
    # Connect to the drone
    connection = connect_to_drone("/dev/ttyACM1")
    set_mode_guided(connection)

    time.sleep(5)
    arm_disarm_drone(connection, 1)

    # Take off to an altitude of 5 meters
    time.sleep(5)
    takeoff_drone(connection, 2)
    time.sleep(10)
    # # Sequentially move the drone according to the specified pattern
    move_forward(connection, 2)
    time.sleep(5)
    # move_right(connection, 10)

    # move_backward(connection, 10)

    # move_left(connection, 10)

    # # Land the drone
    land_drone(connection)

    # # Disarm the drone after landing
    arm_disarm_drone(connection, 0)


if __name__ == "__main__":
    main()
