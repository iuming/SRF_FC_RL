"""
Real-time Control Demo Script

This script demonstrates the real-time control capabilities of the RF cavity system.
"""

import os
import sys
import time

# Add paths
sys.path.append('./src')
sys.path.append('./configs')

def demo_realtime_control():
    """Demonstrate real-time control functionality"""
    print("="*60)
    print("RF Cavity Real-Time Control - Demo")
    print("="*60)
    
    try:
        # Import modules
        from realtime_control import RealTimeControlWrapper
        from config import ENV_CONFIG
        print("✓ Successfully imported real-time control modules")
        
        # Create control wrapper
        wrapper = RealTimeControlWrapper(
            env_config=ENV_CONFIG,
            update_rate_hz=20.0,  # 20 Hz for demo
            buffer_size=100
        )
        print("✓ Created real-time control wrapper")
        
        # Set up callbacks
        def status_callback(message):
            print(f"[STATUS] {message}")
            
        wrapper.set_status_callback(status_callback)
        
        # Start simulation
        print("\n--- Starting Simulation ---")
        wrapper.start_simulation()
        
        # Let it run for a few seconds with no control
        print("Running with no control for 3 seconds...")
        time.sleep(3)
        
        # Show current state
        state = wrapper.get_current_state()
        print(f"\nCurrent state after {state['step_count']} steps:")
        if state['observation'] is not None:
            obs = state['observation']
            print(f"  VC Amplitude: {obs[0]:.4f} MV")
            print(f"  VR Amplitude: {obs[1]:.4f} MV") 
            print(f"  VC Phase: {obs[2]:.2f}°")
            print(f"  Frequency Detuning: {obs[3]:.4f} kHz")
            print(f"  Current Action: {state['action']:.4f}")
            print(f"  Current Reward: {state['reward']:.6f}")
        
        # Enable manual control
        print("\n--- Testing Manual Control ---")
        wrapper.enable_manual_control(True)
        
        # Apply some manual actions
        test_actions = [0.5, -0.3, 0.8, -0.1, 0.0]
        for action in test_actions:
            wrapper.set_manual_action(action)
            print(f"Set manual action to {action:.1f}")
            time.sleep(1)
            
            state = wrapper.get_current_state()
            if state['observation'] is not None:
                obs = state['observation']
                print(f"  Detuning: {obs[3]:.4f} kHz, Reward: {state['reward']:.6f}")
        
        # Get buffer data
        print("\n--- Data Buffer Status ---")
        data = wrapper.get_buffer_data()
        print(f"Buffer contains {len(data['time'])} data points")
        if len(data['time']) > 0:
            print(f"Time range: {data['time'][0]:.3f} - {data['time'][-1]:.3f} seconds")
            print(f"Detuning range: {min(data['frequency_detuning']):.4f} - {max(data['frequency_detuning']):.4f} kHz")
        
        # Pause and resume
        print("\n--- Testing Pause/Resume ---")
        wrapper.pause_simulation()
        time.sleep(1)
        wrapper.resume_simulation()
        time.sleep(1)
        
        # Reset simulation
        print("\n--- Testing Reset ---")
        wrapper.reset_simulation()
        time.sleep(1)
        
        # Final state
        state = wrapper.get_current_state()
        print(f"After reset: Episode {state['episode_count']}, Step {state['step_count']}")
        
        # Stop simulation
        print("\n--- Stopping Simulation ---")
        wrapper.stop_simulation()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("Real-time control interface is working correctly.")
        print("="*60)
        
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_realtime_control()
