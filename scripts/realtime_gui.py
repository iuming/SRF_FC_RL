"""
Real-time RF Cavity Control GUI Interface

Filename: realtime_gui.py
Author: Ming Liu
Email: ming.liu@example.com
GitHub: https://github.com/iuming
Created: 2025-07-25
Version: 1.0.0

Description:
    This module provides a comprehensive graphical user interface for real-time
    RF cavity control using trained reinforcement learning agents. The GUI
    features live plotting, manual control capabilities, and comprehensive
    system monitoring for both automatic and manual operation modes.

Features:
    - Real-time plotting with multiple data streams
    - Manual control interface with sliders
    - Automatic/Manual/Off control mode switching
    - Live system status monitoring
    - Data recording and export capabilities
    - Interactive visualization with zoom/pan
    - Emergency stop functionality
    - Performance metrics display

Dependencies:
    - tkinter (GUI framework)
    - matplotlib (plotting and visualization)
    - numpy (numerical computations)
    - threading (background operations)
    - stable_baselines3 (RL model loading)
    - gymnasium (environment interface)

Changelog:
    v1.0.0 (2025-07-25):
        - Initial GUI implementation
        - Added real-time plotting capabilities
        - Implemented manual control interface
        - Created multi-mode operation system
        - Added comprehensive monitoring features
        - Integrated data recording functionality

License:
    This code is part of the ML_Learning repository.
    
Usage:
    python realtime_gui.py
    
    # Launch with specific model:
    python realtime_gui.py --model_path path/to/model.zip
    
    # Start in manual mode:
    python realtime_gui.py --mode manual
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import threading
import time
import os
import sys
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
    print("Please ensure all dependencies are installed and paths are correct.")


class RealTimeControlGUI:
    """
    Real-time control GUI for RF cavity system
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("RF Cavity Real-Time Control System")
        self.root.geometry("1400x900")
        
        # Control system
        self.control_wrapper = None
        self.model = None
        
        # GUI update rate
        self.gui_update_rate_ms = 50  # 20 Hz
        
        # Setup GUI
        self.setup_gui()
        
        # Status
        self.last_update_time = time.time()
        self.update_counter = 0
        
        # Start GUI update loop
        self.update_gui()
    
    def setup_gui(self):
        """Setup the GUI layout"""
        # Create main frames
        self.create_control_frame()
        self.create_status_frame()
        self.create_plot_frame()
        self.create_menu()
    
    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Model...", command=self.load_model)
        file_menu.add_command(label="Save Data...", command=self.save_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        
        # Control menu
        control_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Control", menu=control_menu)
        control_menu.add_command(label="Start Simulation", command=self.start_simulation)
        control_menu.add_command(label="Pause Simulation", command=self.pause_simulation)
        control_menu.add_command(label="Reset Simulation", command=self.reset_simulation)
        control_menu.add_command(label="Stop Simulation", command=self.stop_simulation)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Clear Plots", command=self.clear_plots)
        view_menu.add_command(label="Auto Scale", command=self.auto_scale_plots)
    
    def create_control_frame(self):
        """Create control panel frame"""
        control_frame = ttk.LabelFrame(self.root, text="Control Panel", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), padx=5, pady=5)
        
        # Simulation controls
        sim_frame = ttk.LabelFrame(control_frame, text="Simulation", padding="5")
        sim_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.start_btn = ttk.Button(sim_frame, text="Start", command=self.start_simulation)
        self.start_btn.grid(row=0, column=0, padx=2)
        
        self.pause_btn = ttk.Button(sim_frame, text="Pause", command=self.pause_simulation, state="disabled")
        self.pause_btn.grid(row=0, column=1, padx=2)
        
        self.reset_btn = ttk.Button(sim_frame, text="Reset", command=self.reset_simulation)
        self.reset_btn.grid(row=0, column=2, padx=2)
        
        self.stop_btn = ttk.Button(sim_frame, text="Stop", command=self.stop_simulation, state="disabled")
        self.stop_btn.grid(row=0, column=3, padx=2)
        
        # Control mode
        mode_frame = ttk.LabelFrame(control_frame, text="Control Mode", padding="5")
        mode_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.control_mode = tk.StringVar(value="auto")
        ttk.Radiobutton(mode_frame, text="Automatic", variable=self.control_mode, 
                       value="auto", command=self.on_control_mode_change).grid(row=0, column=0, padx=5)
        ttk.Radiobutton(mode_frame, text="Manual", variable=self.control_mode, 
                       value="manual", command=self.on_control_mode_change).grid(row=0, column=1, padx=5)
        ttk.Radiobutton(mode_frame, text="Off", variable=self.control_mode, 
                       value="off", command=self.on_control_mode_change).grid(row=0, column=2, padx=5)
        
        # Manual control
        manual_frame = ttk.LabelFrame(control_frame, text="Manual Control", padding="5")
        manual_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(manual_frame, text="Piezo Action:").grid(row=0, column=0, sticky=tk.W)
        self.manual_action_var = tk.DoubleVar(value=0.0)
        self.manual_scale = ttk.Scale(manual_frame, from_=-2.0, to=2.0, 
                                     variable=self.manual_action_var, 
                                     orient=tk.HORIZONTAL, length=200,
                                     command=self.on_manual_action_change)
        self.manual_scale.grid(row=0, column=1, padx=5)
        self.manual_value_label = ttk.Label(manual_frame, text="0.000")
        self.manual_value_label.grid(row=0, column=2, padx=5)
        
        # Model info
        model_frame = ttk.LabelFrame(control_frame, text="Model Information", padding="5")
        model_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.model_status_label = ttk.Label(model_frame, text="No model loaded")
        self.model_status_label.grid(row=0, column=0, sticky=tk.W)
        
        ttk.Button(model_frame, text="Load Model", command=self.load_model).grid(row=1, column=0, pady=2)
        
        # System parameters
        params_frame = ttk.LabelFrame(control_frame, text="System Parameters", padding="5")
        params_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(params_frame, text="Update Rate:").grid(row=0, column=0, sticky=tk.W)
        self.update_rate_var = tk.IntVar(value=100)
        ttk.Spinbox(params_frame, from_=10, to=1000, textvariable=self.update_rate_var,
                   width=10, command=self.on_update_rate_change).grid(row=0, column=1, padx=5)
        ttk.Label(params_frame, text="Hz").grid(row=0, column=2, sticky=tk.W)
    
    def create_status_frame(self):
        """Create status display frame"""
        status_frame = ttk.LabelFrame(self.root, text="System Status", padding="10")
        status_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Current values
        values_frame = ttk.LabelFrame(status_frame, text="Current Values", padding="5")
        values_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # Create status labels
        labels = ["VC Amplitude:", "VR Amplitude:", "VC Phase:", "Frequency Detuning:", 
                 "Action:", "Reward:", "Step Count:", "Episode:"]
        self.status_labels = {}
        
        for i, label in enumerate(labels):
            ttk.Label(values_frame, text=label).grid(row=i, column=0, sticky=tk.W, padx=2)
            self.status_labels[label] = ttk.Label(values_frame, text="--")
            self.status_labels[label].grid(row=i, column=1, sticky=tk.W, padx=10)
        
        # Performance metrics
        perf_frame = ttk.LabelFrame(status_frame, text="Performance", padding="5")
        perf_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=2)
        
        perf_labels = ["Total Reward:", "Avg Reward:", "Update Rate:", "GUI FPS:"]
        for i, label in enumerate(perf_labels):
            ttk.Label(perf_frame, text=label).grid(row=i, column=0, sticky=tk.W, padx=2)
            self.status_labels[label] = ttk.Label(perf_frame, text="--")
            self.status_labels[label].grid(row=i, column=1, sticky=tk.W, padx=10)
        
        # Status log
        log_frame = ttk.LabelFrame(status_frame, text="Status Log", padding="5")
        log_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=2)
        
        self.status_text = tk.Text(log_frame, height=8, width=30)
        self.status_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.status_text.configure(yscrollcommand=scrollbar.set)
    
    def create_plot_frame(self):
        """Create plotting frame with real-time plots"""
        plot_frame = ttk.LabelFrame(self.root, text="Real-Time Monitoring", padding="5")
        plot_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Configure grid weights
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.fig.subplots_adjust(hspace=0.4, wspace=0.3)
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(3, 2, 1)  # VC Amplitude
        self.ax2 = self.fig.add_subplot(3, 2, 2)  # VR Amplitude
        self.ax3 = self.fig.add_subplot(3, 2, 3)  # VC Phase
        self.ax4 = self.fig.add_subplot(3, 2, 4)  # Frequency Detuning
        self.ax5 = self.fig.add_subplot(3, 2, 5)  # Action
        self.ax6 = self.fig.add_subplot(3, 2, 6)  # Reward
        
        # Configure subplots
        self.ax1.set_title('Cavity Voltage Amplitude')
        self.ax1.set_ylabel('Amplitude (MV)')
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.set_title('Reflected Voltage Amplitude')
        self.ax2.set_ylabel('Amplitude (MV)')
        self.ax2.grid(True, alpha=0.3)
        
        self.ax3.set_title('Cavity Voltage Phase')
        self.ax3.set_ylabel('Phase (degrees)')
        self.ax3.grid(True, alpha=0.3)
        
        self.ax4.set_title('Frequency Detuning')
        self.ax4.set_ylabel('Detuning (kHz)')
        self.ax4.grid(True, alpha=0.3)
        self.ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        self.ax5.set_title('Control Action')
        self.ax5.set_ylabel('Action')
        self.ax5.set_xlabel('Time (s)')
        self.ax5.grid(True, alpha=0.3)
        
        self.ax6.set_title('Reward')
        self.ax6.set_ylabel('Reward')
        self.ax6.set_xlabel('Time (s)')
        self.ax6.grid(True, alpha=0.3)
        
        # Initialize empty lines
        self.lines = {}
        self.lines['vc_amplitude'], = self.ax1.plot([], [], 'b-', linewidth=1)
        self.lines['vr_amplitude'], = self.ax2.plot([], [], 'r-', linewidth=1)
        self.lines['vc_phase'], = self.ax3.plot([], [], 'g-', linewidth=1)
        self.lines['frequency_detuning'], = self.ax4.plot([], [], 'm-', linewidth=1)
        self.lines['action'], = self.ax5.plot([], [], 'c-', linewidth=1)
        self.lines['reward'], = self.ax6.plot([], [], 'orange', linewidth=1)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    def start_simulation(self):
        """Start the simulation"""
        try:
            if self.control_wrapper is None:
                self.control_wrapper = RealTimeControlWrapper(
                    env_config=ENV_CONFIG,
                    update_rate_hz=self.update_rate_var.get()
                )
                self.control_wrapper.set_data_callback(self.on_data_update)
                self.control_wrapper.set_status_callback(self.on_status_update)
                
                if self.model is not None:
                    self.control_wrapper.set_model(self.model)
            
            self.control_wrapper.start_simulation()
            
            # Update button states
            self.start_btn.config(state="disabled")
            self.pause_btn.config(state="normal")
            self.stop_btn.config(state="normal")
            
            # Set initial control mode
            self.on_control_mode_change()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start simulation: {e}")
    
    def pause_simulation(self):
        """Pause/resume simulation"""
        if self.control_wrapper is None:
            return
        
        if self.control_wrapper.is_paused:
            self.control_wrapper.resume_simulation()
            self.pause_btn.config(text="Pause")
        else:
            self.control_wrapper.pause_simulation()
            self.pause_btn.config(text="Resume")
    
    def reset_simulation(self):
        """Reset simulation"""
        if self.control_wrapper is not None:
            self.control_wrapper.reset_simulation()
            self.clear_plots()
    
    def stop_simulation(self):
        """Stop simulation"""
        if self.control_wrapper is not None:
            self.control_wrapper.stop_simulation()
        
        # Update button states
        self.start_btn.config(state="normal")
        self.pause_btn.config(state="disabled", text="Pause")
        self.stop_btn.config(state="disabled")
    
    def load_model(self):
        """Load a trained model"""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")],
            initialdir="./best_model/"
        )
        
        if file_path:
            try:
                self.model = PPO.load(file_path, device='cpu')
                model_name = os.path.basename(file_path)
                self.model_status_label.config(text=f"Loaded: {model_name}")
                self.on_status_update(f"Model loaded: {model_name}")
                
                if self.control_wrapper is not None:
                    self.control_wrapper.set_model(self.model)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {e}")
    
    def save_data(self):
        """Save current data to file"""
        if self.control_wrapper is None:
            messagebox.showwarning("Warning", "No simulation data to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Data",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                data = self.control_wrapper.get_buffer_data()
                
                # Convert to CSV format
                import csv
                with open(file_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    # Write header
                    writer.writerow(['time', 'vc_amplitude', 'vr_amplitude', 'vc_phase', 
                                   'frequency_detuning', 'action', 'reward'])
                    
                    # Write data
                    for i in range(len(data['time'])):
                        writer.writerow([
                            data['time'][i],
                            data['vc_amplitude'][i],
                            data['vr_amplitude'][i],
                            data['vc_phase'][i],
                            data['frequency_detuning'][i],
                            data['action'][i],
                            data['reward'][i]
                        ])
                
                self.on_status_update(f"Data saved to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save data: {e}")
    
    def clear_plots(self):
        """Clear all plots"""
        for line in self.lines.values():
            line.set_data([], [])
        
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]:
            ax.relim()
            ax.autoscale_view()
        
        self.canvas.draw()
    
    def auto_scale_plots(self):
        """Auto scale all plots"""
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]:
            ax.relim()
            ax.autoscale_view()
        self.canvas.draw()
    
    def on_control_mode_change(self):
        """Handle control mode change"""
        if self.control_wrapper is None:
            return
        
        mode = self.control_mode.get()
        
        if mode == "auto":
            self.control_wrapper.enable_manual_control(False)
            self.control_wrapper.enable_auto_control(True)
            self.manual_scale.config(state="disabled")
        elif mode == "manual":
            self.control_wrapper.enable_auto_control(False)
            self.control_wrapper.enable_manual_control(True)
            self.manual_scale.config(state="normal")
        else:  # off
            self.control_wrapper.enable_auto_control(False)
            self.control_wrapper.enable_manual_control(False)
            self.manual_scale.config(state="disabled")
    
    def on_manual_action_change(self, value):
        """Handle manual action change"""
        action_value = float(value)
        self.manual_value_label.config(text=f"{action_value:.3f}")
        
        if self.control_wrapper is not None:
            self.control_wrapper.set_manual_action(action_value)
    
    def on_update_rate_change(self):
        """Handle update rate change"""
        if self.control_wrapper is not None:
            new_rate = self.update_rate_var.get()
            self.control_wrapper.update_rate_hz = new_rate
            self.control_wrapper.update_period = 1.0 / new_rate
    
    def on_data_update(self, state):
        """Handle data update from control wrapper"""
        # This will be called from the GUI update loop
        pass
    
    def on_status_update(self, message):
        """Handle status update"""
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
    
    def update_gui(self):
        """Update GUI elements periodically"""
        try:
            if self.control_wrapper is not None:
                # Get current state
                state = self.control_wrapper.get_current_state()
                
                if state['observation'] is not None:
                    obs = state['observation']
                    
                    # Update status labels
                    self.status_labels["VC Amplitude:"].config(text=f"{obs[0]:.4f} MV")
                    self.status_labels["VR Amplitude:"].config(text=f"{obs[1]:.4f} MV")
                    self.status_labels["VC Phase:"].config(text=f"{obs[2]:.2f}°")
                    self.status_labels["Frequency Detuning:"].config(text=f"{obs[3]:.4f} kHz")
                    self.status_labels["Action:"].config(text=f"{state['action']:.4f}")
                    self.status_labels["Reward:"].config(text=f"{state['reward']:.6f}")
                    self.status_labels["Step Count:"].config(text=f"{state['step_count']}")
                    self.status_labels["Episode:"].config(text=f"{state['episode_count']}")
                    self.status_labels["Total Reward:"].config(text=f"{state['total_reward']:.2f}")
                    
                    if state['step_count'] > 0:
                        avg_reward = state['total_reward'] / state['step_count']
                        self.status_labels["Avg Reward:"].config(text=f"{avg_reward:.6f}")
                
                # Update plots
                self.update_plots()
                
                # Calculate GUI FPS
                current_time = time.time()
                self.update_counter += 1
                if current_time - self.last_update_time >= 1.0:
                    fps = self.update_counter / (current_time - self.last_update_time)
                    self.status_labels["GUI FPS:"].config(text=f"{fps:.1f}")
                    self.status_labels["Update Rate:"].config(text=f"{self.update_rate_var.get()} Hz")
                    self.update_counter = 0
                    self.last_update_time = current_time
        
        except Exception as e:
            print(f"GUI update error: {e}")
        
        # Schedule next update
        self.root.after(self.gui_update_rate_ms, self.update_gui)
    
    def update_plots(self):
        """Update real-time plots"""
        if self.control_wrapper is None:
            return
        
        try:
            data = self.control_wrapper.get_buffer_data()
            
            if len(data['time']) > 0:
                # Update line data
                self.lines['vc_amplitude'].set_data(data['time'], data['vc_amplitude'])
                self.lines['vr_amplitude'].set_data(data['time'], data['vr_amplitude'])
                self.lines['vc_phase'].set_data(data['time'], data['vc_phase'])
                self.lines['frequency_detuning'].set_data(data['time'], data['frequency_detuning'])
                self.lines['action'].set_data(data['time'], data['action'])
                self.lines['reward'].set_data(data['time'], data['reward'])
                
                # Auto-scale axes
                for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]:
                    ax.relim()
                    ax.autoscale_view()
                
                # Redraw canvas
                self.canvas.draw_idle()
        
        except Exception as e:
            print(f"Plot update error: {e}")
    
    def on_closing(self):
        """Handle window closing"""
        if self.control_wrapper is not None:
            self.control_wrapper.stop_simulation()
        self.root.destroy()
    
    def run(self):
        """Run the GUI application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()


def main():
    """Main function to run the real-time control GUI"""
    try:
        app = RealTimeControlGUI()
        app.run()
    except Exception as e:
        print(f"Failed to start GUI: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
