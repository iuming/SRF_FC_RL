"""
RF Cavity Control Environment for Reinforcement Learning

Filename: rf_cavity_env.py
Author: Ming Liu
Email: ming.liu@example.com
GitHub: https://github.com/iuming
Created: 2025-07-25
Version: 1.0.0

Description:
    This module implements a comprehensive RF (Radio Frequency) cavity control 
    environment for reinforcement learning applications. The environment simulates 
    a realistic RF cavity system with mechanical modes, beam loading effects, 
    and piezo-based frequency control, designed for training RL agents to minimize 
    frequency detuning.

Features:
    - Physics-based RF cavity simulation
    - Mechanical mode dynamics
    - Beam loading effects
    - Piezo actuator control
    - Real-time rendering capabilities
    - Comprehensive safety checks for numerical stability

Dependencies:
    - gymnasium
    - numpy
    - matplotlib
    - llrflibs (RF simulation library)

Changelog:
    v1.0.0 (2025-07-25):
        - Initial implementation of RF cavity environment
        - Added comprehensive observation and action spaces
        - Implemented physics-based simulation components
        - Added numerical stability and safety checks
        - Integrated real-time rendering capabilities
        - Created modular design for easy configuration

License:
    This code is part of the ML_Learning repository.
    
Usage:
    from rf_cavity_env import RFCavityControlEnv
    
    env = RFCavityControlEnv()
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import gymnasium as gym
import numpy as np
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
from llrflibs.rf_sim import *
from llrflibs.rf_control import *


class RFCavityControlEnv(gym.Env):
    """
    RF Cavity Control Environment
    
    The goal is to control the RF cavity frequency using piezo actuators
    to minimize frequency detuning while maintaining stable operation.
    
    Observation Space:
        - vc_amplitude: Cavity voltage amplitude (MV)
        - vr_amplitude: Reflected voltage amplitude (MV) 
        - vc_phase: Cavity voltage phase (degrees)
        - frequency_detuning: Frequency detuning (kHz)
    
    Action Space:
        - piezo_control: Piezo control signal [-2.0, 2.0]
    
    Reward:
        - Negative absolute frequency detuning (minimize detuning)
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30
    }

    def __init__(self, 
                 render_mode: Optional[str] = None,
                 max_steps: int = 2048 * 16,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RF Cavity Control Environment
        
        Args:
            render_mode: Rendering mode ('human', 'rgb_array', or None)
            max_steps: Maximum number of steps per episode
            config: Configuration dictionary for environment parameters
        """
        super().__init__()
        
        # Episode configuration
        self.max_steps = max_steps
        self.dt = 2 * np.pi / (self.max_steps - 1)
        
        # Spaces definition
        self.action_space = gym.spaces.Box(
            low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.array([-1000.0, -1000.0, -360.0, -1000.0]),
            high=np.array([1000.0, 1000.0, 360.0, 1000.0]),
            shape=(4,),
            dtype=np.float32
        )

        # Action history for potential future use
        self.action_history = [np.zeros(1) for _ in range(3)]

        # Rendering setup
        self.render_mode = render_mode
        self.fig = None
        self.line = None
        self.history_t = []
        self.history_ft = []

        # Initialize RF system parameters
        self._init_rf_parameters(config)
        
        # Initialize simulation components
        self._init_simulation_components()

    def _init_rf_parameters(self, config: Optional[Dict[str, Any]] = None):
        """Initialize RF system parameters"""
        if config is None:
            config = {}
            
        # General parameters
        self.Ts = config.get('sampling_time', 1e-6)
        self.t_fill = config.get('fill_time', 510)
        self.t_flat = config.get('flat_time', 1300)

        # RF source parameters
        self.fsrc = config.get('source_frequency', -460)
        self.Asrc = config.get('source_amplitude', 1)
        self.pha_src = 0

        # I/Q modulator parameters
        self.pulsed = config.get('pulsed_mode', True)
        self.buf_size = config.get('buffer_size', 2048 * 8)
        self.base_pul = np.zeros(self.buf_size, dtype=complex)
        self.base_cw = 1
        self.base_pul[:self.t_flat] = 1.0
        self.buf_id = 0

        # Amplifier parameters
        self.gain_dB = config.get('amplifier_gain_db', 20 * np.log10(12e6))

        # Cavity parameters
        self.mech_modes = config.get('mechanical_modes', {
            'f': [280, 341, 460, 487, 618],
            'Q': [40, 20, 50, 80, 100],
            'K': [2, 0.8, 2, 0.6, 0.2]
        })
        self.f0 = config.get('cavity_frequency', 1.3e9)
        self.beta = config.get('coupling_beta', 1e4)
        self.roQ = config.get('cavity_roQ', 1036)
        self.QL = config.get('loaded_q', 3e6)
        self.RL = 0.5 * self.roQ * self.QL
        self.wh = np.pi * self.f0 / self.QL
        self.ib = config.get('beam_current', 0.008)
        self.dw0 = 2 * np.pi * 0

        # Beam loading
        self.beam_pul = np.zeros(self.buf_size, dtype=complex)
        self.beam_cw = 0
        self.beam_pul[self.t_fill:self.t_flat] = self.ib

        # Simulation parameters
        self.sim_len = config.get('simulation_length', 2048 * 500)
        self.pul_len = config.get('pulse_length', 2048 * 20)

    def _init_simulation_components(self):
        """Initialize mechanical mode simulation components"""
        status, Am, Bm, Cm, Dm = cav_ss_mech(self.mech_modes)
        status, Ad, Bd, Cd, Dd, _ = ss_discrete(
            Am, Bm, Cm, Dm,
            Ts=self.Ts,
            method='zoh',
            plot=False,
            plot_pno=10000
        )
        
        self.state_m = np.matrix(np.zeros(Bd.shape))
        self.state_vc = 0.0
        self.Ad = Ad
        self.Bd = Bd
        self.Cd = Cd
        self.Dd = Dd
        
        # Initialize state variables
        self.dw = 0
        self.prev_dw = 0

    def _simulate_rf_source(self):
        """Simulate RF source"""
        pha = self.pha_src + 2.0 * np.pi * self.fsrc * self.Ts
        return self.Asrc * np.exp(1j * pha), pha

    def _simulate_iq_modulator(self, sig_in):
        """Simulate I/Q modulator"""
        if self.pulsed:
            sig_out = sig_in * self.base_pul[
                self.buf_id if self.buf_id < len(self.base_pul) else -1
            ]
        else:
            sig_out = sig_in * self.base_cw
        return sig_out

    def _simulate_amplifier(self, sig_in):
        """Simulate RF amplifier"""
        return sig_in * 10.0 ** (self.gain_dB / 20.0)

    def _simulate_cavity(self, vf_step, detuning):
        """Simulate RF cavity with mechanical modes"""
        if self.pulsed:
            vb = -self.RL * self.beam_pul[
                self.buf_id if self.buf_id < len(self.beam_pul) else -1
            ]
        else:
            vb = self.beam_cw

        status, vc, vr, dw, state_m = sim_scav_step(
            self.wh,
            self.dw,
            detuning,
            vf_step,
            vb,
            self.state_vc,
            self.Ts,
            beta=self.beta,
            state_m0=self.state_m,
            Am=self.Ad,
            Bm=self.Bd,
            Cm=self.Cd,
            Dm=self.Dd,
            mech_exe=True
        )
        
        self.state_vc = vc
        return vc, vr, dw, self.state_vc, state_m

    def _build_observation(self, vc, vr, dw):
        """Build observation vector with safety checks"""
        # Extract values and handle matrix types
        vc_abs = abs(vc) if vc is not None else 0.0
        vr_abs = abs(vr) if vr is not None else 0.0
        vc_angle = np.angle(vc) if vc is not None else 0.0
        dw_val = dw if dw is not None else 0.0
        
        # Handle matrix types
        if isinstance(vc_abs, np.matrix):
            vc_abs = vc_abs.item()
        if isinstance(vr_abs, np.matrix):
            vr_abs = vr_abs.item()
        if isinstance(vc_angle, np.matrix):
            vc_angle = vc_angle.item()
        if isinstance(dw_val, np.matrix):
            dw_val = dw_val.item()
        
        # Ensure all values are valid numbers
        vc_abs = np.nan_to_num(vc_abs, nan=0.0, posinf=1e3, neginf=-1e3)
        vr_abs = np.nan_to_num(vr_abs, nan=0.0, posinf=1e3, neginf=-1e3)
        vc_angle = np.nan_to_num(vc_angle, nan=0.0, posinf=np.pi, neginf=-np.pi)
        dw_val = np.nan_to_num(dw_val, nan=0.0, posinf=1e9, neginf=-1e9)
        
        # Build observation vector with proper scaling and clipping
        obs_vc = np.clip(float(vc_abs * 1e-6), -1000.0, 1000.0)
        obs_vr = np.clip(float(vr_abs * 1e-6), -1000.0, 1000.0)
        obs_angle = np.clip(float(vc_angle * 180 / np.pi), -360.0, 360.0)
        obs_dw = np.clip(float(dw_val * 1e-3), -1000.0, 1000.0)
        
        return np.array([obs_vc, obs_vr, obs_angle, obs_dw], dtype=np.float32)

    def _calculate_reward(self, observation):
        """Calculate reward based on frequency detuning"""
        # Reward is negative absolute frequency detuning
        reward = -np.abs(observation[3])
        reward = np.nan_to_num(reward, nan=-1.0, posinf=-1.0, neginf=-1.0)
        reward = np.clip(reward, -1000.0, 1000.0)
        return reward

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset episode variables
        self.t = 0.0
        self.step_count = 0
        self.history_t = []
        self.history_ft = []
        
        # Reset RF system state
        self.pha_src = 0
        self.buf_id = 0
        self.state_m = np.matrix(np.zeros(self.Bd.shape))
        self.state_vc = 0.0
        self.dw = 0
        self.prev_dw = 0

        # Run initial simulation step
        S0, self.pha_src = self._simulate_rf_source()
        S1 = self._simulate_iq_modulator(S0)
        S2 = self._simulate_amplifier(S1)
        dw_micr = 2.0 * np.pi * np.random.randn() * 10
        dw_piezo = 0
        vc, vr, self.dw, self.state_vc, self.state_m = self._simulate_cavity(
            S2, dw_piezo + dw_micr
        )

        if isinstance(vc, np.matrix):
            vc = vc.item()

        # Build initial observation
        observation = self._build_observation(vc, vr, self.dw)
        
        return observation, {}

    def step(self, action):
        """Execute one environment step"""
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Update RF system
        self.S0, self.pha_src = self._simulate_rf_source()

        # Update buffer index for pulsed mode
        if self.pulsed:
            self.buf_id += 1
            if self.buf_id >= self.pul_len:
                self.buf_id = 0

        # RF signal chain simulation
        self.S1 = self._simulate_iq_modulator(self.S0)
        self.S2 = self._simulate_amplifier(self.S1)
        
        # Add microphonics and piezo control
        self.dw_micr = 2.0 * np.pi * np.random.randn() * 0  # No microphonics for now
        dw_piezo = 2 * np.pi * action[0] * 1e4
        
        # Simulate cavity response
        self.vc, self.vr, self.dw, self.state_vc, self.state_m = self._simulate_cavity(
            self.S2, dw_piezo + self.dw_micr
        )

        # Update action history
        self.action_history.pop(0)
        self.action_history.append(action)
        
        # Update previous detuning
        self.prev_dw = self.dw

        # Build observation
        obs = self._build_observation(self.vc, self.vr, self.dw)

        # Record data for rendering
        self.history_t.append(self.t)
        self.history_ft.append(obs[0].item())

        # Calculate reward
        reward = self._calculate_reward(obs)

        # Update time and step count
        self.t += self.dt
        self.step_count += 1

        # Check termination conditions
        terminated = False
        truncated = self.step_count >= self.max_steps
        info = {
            'vc_amplitude': obs[0],
            'vr_amplitude': obs[1], 
            'vc_phase': obs[2],
            'frequency_detuning': obs[3]
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment"""
        if self.render_mode is None:
            return

        # Create or update plot
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(12, 8))
            
            # Create subplots
            self.ax1 = self.fig.add_subplot(221)
            self.ax2 = self.fig.add_subplot(222)
            self.ax3 = self.fig.add_subplot(223)
            self.ax4 = self.fig.add_subplot(224)
            
            self.line1, = self.ax1.plot([], [], 'b-')
            self.line2, = self.ax2.plot([], [], 'r-')
            self.line3, = self.ax3.plot([], [], 'g-')
            self.line4, = self.ax4.plot([], [], 'm-')
            
            self.ax1.set_title('Cavity Voltage Amplitude')
            self.ax1.set_ylabel('Amplitude (MV)')
            
            self.ax2.set_title('Reflected Voltage Amplitude')
            self.ax2.set_ylabel('Amplitude (MV)')
            
            self.ax3.set_title('Cavity Voltage Phase')
            self.ax3.set_ylabel('Phase (degrees)')
            
            self.ax4.set_title('Frequency Detuning')
            self.ax4.set_xlabel('Time')
            self.ax4.set_ylabel('Detuning (kHz)')

        # Update data (sample for performance if needed)
        if len(self.history_t) > 1000:
            # Sample data for better performance
            sample_rate = len(self.history_t) // 1000
            t_sampled = self.history_t[::sample_rate]
            ft_sampled = self.history_ft[::sample_rate]
        else:
            t_sampled = self.history_t
            ft_sampled = self.history_ft

        # Update plots (placeholder - you would need actual data for each subplot)
        self.line1.set_data(t_sampled, ft_sampled)
        
        # Update axis limits
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.relim()
            ax.autoscale_view()

        if self.render_mode == 'human':
            plt.draw()
            plt.pause(0.001)
        elif self.render_mode == 'rgb_array':
            self.fig.canvas.draw()
            img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return img

    def close(self):
        """Close the environment"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None


# Alias for backward compatibility
SinEnv = RFCavityControlEnv
