#!/usr/bin/env python3
"""
Simple NICO Motor Test - Core Motors Only
Tests only the available motors without hand initialization
"""

import sys
import time
import traceback

# Add the nicomotion library path
#sys.path.insert(0, '/home/amr/catkin_ws/src/NICO-software/api/src/nicomotion/scripts')

try:
    from nicomotion.Motion import Motion
    print("✓ Successfully imported Motion class")
except ImportError as e:
    print(f"✗ Failed to import Motion class: {e}")
    sys.exit(1)

class SimpleMotorTest:
    def __init__(self):
        self.motion = None
        self.config_path = "/home/hb/NICO-software/api/test_config.json"
        
    def initialize_robot(self):
        """Initialize connection to the robot hardware"""
        print("Initializing robot connection...")
        try:
            # Initialize without hand current monitoring to avoid hand initialization
            self.motion = Motion(motorConfig=self.config_path, vrep=False, 
                               ignoreMissing=True, monitorHandCurrents=False)
            print("✓ Robot connection established successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to initialize robot: {e}")
            traceback.print_exc()
            return False
    
    def test_available_motors(self):
        """Test the available motors based on the successful initialization"""
        print("\n" + "="*50)
        print("TESTING AVAILABLE MOTORS")
        print("="*50)
        
        # Test motors that should be available based on the config output
        test_motors = [
            ("head_z", [-45, 45, 0], "Head Yaw"),
            ("head_y", [30, -30, 0], "Head Pitch"),
            ("r_shoulder_y", [30, -30, 0], "Right Shoulder Pitch"),
            ("r_shoulder_z", [30, -30, 0], "Right Shoulder Roll"),
            ("r_arm_x", [30, -30, 0], "Right Arm Twist"),
            ("r_elbow_y", [30, -30, 0], "Right Elbow"),
            ("l_shoulder_y", [30, -30, 0], "Left Shoulder Pitch"),
            ("l_shoulder_z", [30, -30, 0], "Left Shoulder Roll"),
            ("l_arm_x", [30, -30, 0], "Left Arm Twist"),
            ("l_elbow_y", [30, -30, 0], "Left Elbow"),
        ]
        
        working_motors = []
        failed_motors = []
        
        for motor_name, angles, description in test_motors:
            print(f"\nTesting {description} ({motor_name})...")
            try:
                # Check if motor exists
                current_pos = self.motion.getAngle(motor_name)
                print(f"  Current position: {current_pos:.1f}°")
                
                # Test first angle
                target_angle = angles[0]
                print(f"  Moving to {target_angle}°...")
                self.motion.setAngle(motor_name, target_angle, 0.2)
                time.sleep(1.5)
                
                # Check position
                actual_pos = self.motion.getAngle(motor_name)
                print(f"  Actual position: {actual_pos:.1f}°")
                
                # Return to center
                self.motion.setAngle(motor_name, 0.0, 0.2)
                time.sleep(1.0)
                
                working_motors.append((motor_name, description))
                print(f"  ✓ {description} working")
                
            except Exception as e:
                failed_motors.append((motor_name, description, str(e)))
                print(f"  ✗ {description} failed: {e}")
        
        # Report results
        print("\n" + "="*60)
        print("MOTOR TEST RESULTS")
        print("="*60)
        
        print(f"\n✓ WORKING MOTORS ({len(working_motors)}):")
        for motor_name, description in working_motors:
            print(f"  - {description} ({motor_name})")
        
        print(f"\n✗ FAILED MOTORS ({len(failed_motors)}):")
        for motor_name, description, error in failed_motors:
            print(f"  - {description} ({motor_name}): {error}")
        
        return len(working_motors) > 0
    
    def shutdown(self):
        """Safely shutdown the robot connection"""
        print("\nShutting down robot connection...")
        try:
            if self.motion:
                print("✓ Robot connection closed safely")
        except Exception as e:
            print(f"Warning during shutdown: {e}")

def main():
    test = SimpleMotorTest()
    
    try:
        if test.initialize_robot():
            success = test.test_available_motors()
            if success:
                print("\n🎉 Motor testing completed!")
            else:
                print("\n❌ No motors were working")
        else:
            print("\n❌ Failed to initialize robot")
            
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        traceback.print_exc()
    finally:
        test.shutdown()

if __name__ == "__main__":
    main()
