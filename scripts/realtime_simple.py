"""
Simple Real-time RF Cavity Control Demo

Filename: realtime_simple.py
Author: Ming Liu
Email: ming.liu@example.com
GitHub: https://github.com/iuming
Created: 2025-07-25
Version: 1.0.0

Description:
    This script provides a simple command-line demonstration of real-time
    RF cavity control using trained reinforcement learning agents. It serves
    as a lightweight alternative to the full GUI interface and is ideal for
    quick testing and validation of trained models.

Features:
    - Command-line real-time control interface
    - Live system status monitoring
    - Trained model integration
    - Basic control mode switching
    - Performance metrics display
    - Lightweight operation
    - Quick model validation

Dependencies:
    - stable_baselines3 (RL model loading)
    - gymnasium (environment interface)
    - numpy (numerical computations)
    - time (timing operations)
    - threading (background operations)

Changelog:
    v1.0.0 (2025-07-25):
        - Initial simple interface implementation
        - Added command-line control capabilities
        - Implemented basic monitoring features
        - Created lightweight model testing
        - Added performance metrics display

License:
    This code is part of the ML_Learning repository.
    
Usage:
    python realtime_simple.py
    
    # Run with specific model:
    python realtime_simple.py --model_path path/to/model.zip
    
    # Set custom duration:
    python realtime_simple.py --duration 60
"""

import os
import sys
import time
import threading
import numpy as np
from typing import Optional

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'configs'))

try:
    from realtime_control import RealTimeControlWrapper
    from config import ENV_CONFIG
    from stable_baselines3 import PPO
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class SimpleRealTimeControl:
    """
    Simple command-line real-time control interface
    """
    
    def __init__(self):
        self.control_wrapper = None
        self.model = None
        self.running = False
        
        print("="*60)
        print("RF Cavity Real-Time Control - Simple Interface")
        print("="*60)
    
    def load_model(self, model_path: str) -> bool:
        """Load a trained model"""
        try:
            self.model = PPO.load(model_path, device='cpu')
            print(f"✓ Model loaded: {model_path}")
            return True
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            return False
    
    def start_control(self):
        """Start the real-time control system"""
        try:
            # Initialize control wrapper
            self.control_wrapper = RealTimeControlWrapper(
                env_config=ENV_CONFIG,
                update_rate_hz=50.0  # 50 Hz update rate
            )
            
            # Set callbacks
            self.control_wrapper.set_status_callback(self.status_callback)
            
            # Set model if available
            if self.model is not None:
                self.control_wrapper.set_model(self.model)
                print("✓ Automatic control enabled")
            else:
                print("! No model loaded - manual control only")
            
            # Start simulation
            self.control_wrapper.start_simulation()
            self.running = True
            
            print("\nReal-time control started!")
            print("Commands:")
            print("  a - Enable automatic control")
            print("  m - Enable manual control")
            print("  o - Turn off control")
            print("  p - Pause/Resume simulation")
            print("  r - Reset simulation")
            print("  s - Show status")
            print("  q - Quit")
            print()
            
            # Start command input thread
            command_thread = threading.Thread(target=self.command_loop, daemon=True)
            command_thread.start()
            
            # Start monitoring loop
            self.monitoring_loop()
            
        except Exception as e:
            print(f"✗ Failed to start control system: {e}")
    
    def status_callback(self, message: str):
        """Handle status updates"""
        print(f"[STATUS] {message}")
    
    def command_loop(self):
        """Handle user commands"""
        while self.running:
            try:
                command = input().strip().lower()
                
                if command == 'q':
                    self.running = False
                    break
                elif command == 'a':
                    if self.model is not None:
                        self.control_wrapper.enable_auto_control(True)
                        self.control_wrapper.enable_manual_control(False)
                        print("✓ Automatic control enabled")
                    else:
                        print("✗ No model loaded")
                elif command == 'm':
                    self.control_wrapper.enable_auto_control(False)
                    self.control_wrapper.enable_manual_control(True)
                    print("✓ Manual control enabled")
                    self.manual_control_loop()
                elif command == 'o':
                    self.control_wrapper.enable_auto_control(False)
                    self.control_wrapper.enable_manual_control(False)
                    print("✓ Control disabled")
                elif command == 'p':
                    if self.control_wrapper.is_paused:
                        self.control_wrapper.resume_simulation()
                        print("✓ Simulation resumed")
                    else:
                        self.control_wrapper.pause_simulation()
                        print("✓ Simulation paused")
                elif command == 'r':
                    self.control_wrapper.reset_simulation()
                    print("✓ Simulation reset")
                elif command == 's':
                    self.show_status()
                else:
                    print("Unknown command. Use 'q' to quit.")
                    
            except EOFError:
                break
            except Exception as e:
                print(f"Command error: {e}")
    
    def manual_control_loop(self):
        """Handle manual control input"""
        print("\nManual Control Mode")
        print("Enter action values (-2.0 to 2.0), or 'exit' to return:")
        
        while self.running and self.control_wrapper.manual_control_enabled:
            try:
                user_input = input("Action: ").strip()
                
                if user_input.lower() == 'exit':
                    break
                
                try:
                    action = float(user_input)
                    action = np.clip(action, -2.0, 2.0)
                    self.control_wrapper.set_manual_action(action)
                    print(f"✓ Action set to {action:.3f}")
                except ValueError:
                    print("✗ Invalid number. Enter a value between -2.0 and 2.0")
                    
            except EOFError:
                break
        
        print("Exiting manual control mode")
    
    def show_status(self):
        """Display current system status"""
        if self.control_wrapper is None:
            print("✗ Control system not started")
            return
        
        state = self.control_wrapper.get_current_state()
        
        print("\n" + "="*50)
        print("SYSTEM STATUS")
        print("="*50)
        
        if state['observation'] is not None:
            obs = state['observation']
            print(f"VC Amplitude:       {obs[0]:.4f} MV")
            print(f"VR Amplitude:       {obs[1]:.4f} MV")
            print(f"VC Phase:           {obs[2]:.2f}°")
            print(f"Frequency Detuning: {obs[3]:.4f} kHz")
        
        print(f"Current Action:     {state['action']:.4f}")
        print(f"Current Reward:     {state['reward']:.6f}")
        print(f"Step Count:         {state['step_count']}")
        print(f"Episode:            {state['episode_count']}")
        print(f"Total Reward:       {state['total_reward']:.2f}")
        
        if state['step_count'] > 0:
            avg_reward = state['total_reward'] / state['step_count']
            print(f"Average Reward:     {avg_reward:.6f}")
        
        print(f"Running:            {state['is_running']}")
        print(f"Paused:             {state['is_paused']}")
        print(f"Manual Control:     {state['manual_control']}")
        print(f"Auto Control:       {state['auto_control']}")
        print("="*50)
    
    def monitoring_loop(self):
        """Main monitoring loop"""
        try:
            last_status_time = time.time()
            
            while self.running:
                current_time = time.time()
                
                # Show periodic status (every 5 seconds)
                if current_time - last_status_time >= 5.0:
                    if self.control_wrapper is not None:
                        state = self.control_wrapper.get_current_state()
                        if state['observation'] is not None:
                            obs = state['observation']
                            print(f"[MONITOR] Step {state['step_count']:5d} | "
                                 f"Detuning: {obs[3]:8.4f} kHz | "
                                 f"Action: {state['action']:6.3f} | "
                                 f"Reward: {state['reward']:8.6f}")
                    
                    last_status_time = current_time
                
                time.sleep(0.1)  # 10 Hz monitoring
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print("\nCleaning up...")
        if self.control_wrapper is not None:
            self.control_wrapper.stop_simulation()
        print("✓ Real-time control stopped")


def main():
    """Main function"""
    # Try to find and load a model
    model_paths = [
        "./best_model/best_model.zip",
        "../best_model/best_model.zip",
        "./models/ppo_rf_cavity_final.zip",
        "./ppo_sin_final.zip"
    ]
    
    control = SimpleRealTimeControl()
    
    # Try to load a model
    for path in model_paths:
        if os.path.exists(path):
            if control.load_model(path):
                break
    
    # Start control system
    try:
        control.start_control()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
