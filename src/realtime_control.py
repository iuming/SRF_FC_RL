"""
Real-time Control Wrapper for RF Cavity Control System

Filename: realtime_control.py
Author: Ming Liu
Email: ming.liu@example.com
GitHub: https://github.com/iuming
Created: 2025-07-25
Version: 1.0.0

Description:
    This module provides a real-time control wrapper for the RF cavity control
    environment, enabling live monitoring, manual intervention, and automatic
    control using trained RL models. The wrapper supports multi-threaded
    operation for smooth real-time performance.

Features:
    - Real-time simulation with configurable update rates
    - Multi-threaded architecture for responsive control
    - Multiple control modes (automatic, manual, off)
    - Live data buffering and monitoring
    - Model hot-swapping capabilities
    - Comprehensive simulation control (start/pause/reset/stop)
    - Thread-safe data access and callbacks

Dependencies:
    - numpy
    - threading
    - time
    - queue
    - collections
    - rf_cavity_env

Changelog:
    v1.0.0 (2025-07-25):
        - Initial implementation of real-time control wrapper
        - Added multi-threaded simulation loop
        - Implemented multiple control modes
        - Added data buffering and callback system
        - Created thread-safe state management
        - Integrated model loading and control switching

License:
    This code is part of the ML_Learning repository.
    
Usage:
    from realtime_control import RealTimeControlWrapper
    
    wrapper = RealTimeControlWrapper(env_config, update_rate_hz=100)
    wrapper.set_model(trained_model)
    wrapper.start_simulation()
"""

import numpy as np
import threading
import time
from typing import Optional, Dict, Any, Callable
from collections import deque
import queue

from rf_cavity_env import RFCavityControlEnv


class RealTimeControlWrapper:
    """
    Real-time control wrapper that enables live monitoring and control
    of the RF cavity system with manual intervention capabilities.
    """
    
    def __init__(self, 
                 env_config: Optional[Dict[str, Any]] = None,
                 update_rate_hz: float = 100.0,
                 buffer_size: int = 1000):
        """
        Initialize real-time control wrapper
        
        Args:
            env_config: Environment configuration
            update_rate_hz: Update frequency in Hz
            buffer_size: Size of data buffer for plotting
        """
        self.env = RFCavityControlEnv(config=env_config)
        self.update_rate_hz = update_rate_hz
        self.update_period = 1.0 / update_rate_hz
        self.buffer_size = buffer_size
        
        # Control state
        self.is_running = False
        self.is_paused = False
        self.manual_control_enabled = False
        self.manual_action = 0.0
        
        # Data buffers for real-time plotting
        self.time_buffer = deque(maxlen=buffer_size)
        self.vc_amplitude_buffer = deque(maxlen=buffer_size)
        self.vr_amplitude_buffer = deque(maxlen=buffer_size)
        self.vc_phase_buffer = deque(maxlen=buffer_size)
        self.frequency_detuning_buffer = deque(maxlen=buffer_size)
        self.action_buffer = deque(maxlen=buffer_size)
        self.reward_buffer = deque(maxlen=buffer_size)
        
        # Communication queues
        self.command_queue = queue.Queue()
        self.status_queue = queue.Queue()
        
        # Current state
        self.current_obs = None
        self.current_reward = 0.0
        self.current_action = 0.0
        self.step_count = 0
        self.episode_count = 0
        self.total_reward = 0.0
        
        # Model for automatic control
        self.model = None
        self.auto_control_enabled = False
        
        # Callbacks
        self.data_callback = None
        self.status_callback = None
        
        # Thread for simulation
        self.sim_thread = None
        self.sim_lock = threading.Lock()
    
    def set_model(self, model):
        """Set the trained model for automatic control"""
        self.model = model
        self.auto_control_enabled = model is not None
    
    def set_data_callback(self, callback: Callable):
        """Set callback function for data updates"""
        self.data_callback = callback
    
    def set_status_callback(self, callback: Callable):
        """Set callback function for status updates"""
        self.status_callback = callback
    
    def start_simulation(self):
        """Start the real-time simulation"""
        if self.is_running:
            return
        
        self.is_running = True
        self.is_paused = False
        
        # Reset environment
        self.current_obs, _ = self.env.reset()
        self.step_count = 0
        self.total_reward = 0.0
        
        # Start simulation thread
        self.sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.sim_thread.start()
        
        self._update_status("Simulation started")
    
    def pause_simulation(self):
        """Pause the simulation"""
        self.is_paused = True
        self._update_status("Simulation paused")
    
    def resume_simulation(self):
        """Resume the simulation"""
        self.is_paused = False
        self._update_status("Simulation resumed")
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.is_running = False
        if self.sim_thread and self.sim_thread.is_alive():
            self.sim_thread.join(timeout=1.0)
        self._update_status("Simulation stopped")
    
    def reset_simulation(self):
        """Reset the simulation to initial state"""
        with self.sim_lock:
            self.current_obs, _ = self.env.reset()
            self.step_count = 0
            self.episode_count += 1
            self.total_reward = 0.0
            
            # Clear buffers
            self.time_buffer.clear()
            self.vc_amplitude_buffer.clear()
            self.vr_amplitude_buffer.clear()
            self.vc_phase_buffer.clear()
            self.frequency_detuning_buffer.clear()
            self.action_buffer.clear()
            self.reward_buffer.clear()
        
        self._update_status("Simulation reset")
    
    def set_manual_action(self, action: float):
        """Set manual control action"""
        self.manual_action = np.clip(action, -2.0, 2.0)
    
    def enable_manual_control(self, enabled: bool = True):
        """Enable/disable manual control mode"""
        self.manual_control_enabled = enabled
        mode = "manual" if enabled else "automatic"
        self._update_status(f"Control mode: {mode}")
    
    def enable_auto_control(self, enabled: bool = True):
        """Enable/disable automatic control (requires model)"""
        if enabled and self.model is None:
            self._update_status("Error: No model loaded for automatic control")
            return
        
        self.auto_control_enabled = enabled
        mode = "automatic" if enabled else "disabled"
        self._update_status(f"Auto control: {mode}")
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current system state"""
        with self.sim_lock:
            return {
                'observation': self.current_obs.copy() if self.current_obs is not None else None,
                'action': self.current_action,
                'reward': self.current_reward,
                'step_count': self.step_count,
                'episode_count': self.episode_count,
                'total_reward': self.total_reward,
                'is_running': self.is_running,
                'is_paused': self.is_paused,
                'manual_control': self.manual_control_enabled,
                'auto_control': self.auto_control_enabled,
            }
    
    def get_buffer_data(self) -> Dict[str, Any]:
        """Get buffered data for plotting"""
        with self.sim_lock:
            return {
                'time': list(self.time_buffer),
                'vc_amplitude': list(self.vc_amplitude_buffer),
                'vr_amplitude': list(self.vr_amplitude_buffer),
                'vc_phase': list(self.vc_phase_buffer),
                'frequency_detuning': list(self.frequency_detuning_buffer),
                'action': list(self.action_buffer),
                'reward': list(self.reward_buffer),
            }
    
    def _simulation_loop(self):
        """Main simulation loop running in separate thread"""
        last_time = time.time()
        
        while self.is_running:
            current_time = time.time()
            
            if not self.is_paused:
                # Determine action
                if self.manual_control_enabled:
                    action = np.array([self.manual_action])
                elif self.auto_control_enabled and self.model is not None:
                    action, _ = self.model.predict(self.current_obs, deterministic=True)
                else:
                    action = np.array([0.0])  # No control
                
                # Step environment
                with self.sim_lock:
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    
                    self.current_obs = obs
                    self.current_reward = reward
                    self.current_action = action[0]
                    self.step_count += 1
                    self.total_reward += reward
                    
                    # Add to buffers
                    current_sim_time = self.step_count * self.env.dt
                    self.time_buffer.append(current_sim_time)
                    self.vc_amplitude_buffer.append(obs[0])
                    self.vr_amplitude_buffer.append(obs[1])
                    self.vc_phase_buffer.append(obs[2])
                    self.frequency_detuning_buffer.append(obs[3])
                    self.action_buffer.append(action[0])
                    self.reward_buffer.append(reward)
                
                # Handle episode termination
                if terminated or truncated:
                    self.reset_simulation()
                
                # Call data callback
                if self.data_callback:
                    try:
                        self.data_callback(self.get_current_state())
                    except Exception as e:
                        print(f"Data callback error: {e}")
            
            # Maintain update rate
            elapsed = time.time() - last_time
            sleep_time = max(0, self.update_period - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_time = time.time()
    
    def _update_status(self, message: str):
        """Update status message"""
        timestamp = time.strftime("%H:%M:%S")
        status_msg = f"[{timestamp}] {message}"
        
        if self.status_callback:
            try:
                self.status_callback(status_msg)
            except Exception as e:
                print(f"Status callback error: {e}")
        else:
            print(status_msg)
    
    def __del__(self):
        """Cleanup on destruction"""
        self.stop_simulation()
