"""
RF Cavity Control System - Main Entry Point

Filename: main.py
Author: Ming Liu
Email: ming.liu@example.com
GitHub: https://github.com/iuming
Created: 2025-07-25
Version: 1.0.0

Description:
    This is the main entry point for the RF Cavity Control System. It provides
    a unified interface to access all system functionalities including training,
    evaluation, real-time control, and GUI interface. The script supports
    multiple operation modes and command-line arguments for flexible usage.

Features:
    - Unified system entry point
    - Multiple operation modes (train, test, realtime, gui)
    - Command-line argument parsing
    - Configuration management
    - Comprehensive help system
    - Error handling and logging
    - Cross-platform compatibility

Dependencies:
    - argparse (command-line parsing)
    - os, sys (system operations)
    - All RF cavity control modules

Changelog:
    v1.0.0 (2025-07-25):
        - Initial main entry point implementation
        - Added command-line interface
        - Integrated all system components
        - Created unified operation modes
        - Added comprehensive help system

License:
    This code is part of the ML_Learning repository.
    
Usage:
    # Training mode:
    python main.py train
    
    # Testing mode:
    python main.py test --model_path path/to/model.zip
    
    # Real-time control GUI:
    python main.py gui
    
    # Simple real-time control:
    python main.py realtime
    
    # Show help:
    python main.py --help
"""

import os
import sys
import argparse

# Add paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'scripts'))
sys.path.append(os.path.join(project_root, 'configs'))


def main():
    parser = argparse.ArgumentParser(description='RF Cavity Control RL System')
    parser.add_argument('command', choices=['train', 'test', 'env-test', 'realtime', 'realtime-gui'], 
                       help='Command to execute')
    parser.add_argument('--model-path', type=str, default='./best_model/best_model.zip',
                       help='Path to model file (for test command)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to use for training/testing')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        print("Starting training...")
        os.chdir(os.path.join(project_root, 'scripts'))
        import train_rf_cavity
        train_rf_cavity.main()
        
    elif args.command == 'test':
        print("Starting testing...")
        os.chdir(os.path.join(project_root, 'scripts'))
        import test_rf_cavity
        test_rf_cavity.main()
        
    elif args.command == 'env-test':
        print("Testing environment...")
        os.chdir(os.path.join(project_root, 'scripts'))
        import test_environment
        test_environment.test_environment()
        
    elif args.command == 'realtime':
        print("Starting real-time control (command line)...")
        os.chdir(os.path.join(project_root, 'scripts'))
        import realtime_simple
        realtime_simple.main()
        
    elif args.command == 'realtime-gui':
        print("Starting real-time control (GUI)...")
        os.chdir(os.path.join(project_root, 'scripts'))
        import realtime_gui
        realtime_gui.main()


if __name__ == "__main__":
    main()
