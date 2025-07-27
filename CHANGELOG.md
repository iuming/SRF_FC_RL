# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-07-25

### Added
- Initial release of SRF FC RL (Superconducting RadioFrequency cavity Frequency Control by Reinforcement Learning)
- PPO-based reinforcement learning agent for RF cavity frequency control
- Physics-based RF cavity environment simulation with:
  - 4D observation space (cavity voltage amplitude, reflected voltage amplitude, cavity voltage phase, frequency detuning)
  - 1D continuous action space (piezo control signal)
  - Realistic cavity dynamics including mechanical modes and beam loading
- Real-time control interfaces:
  - Command line interface with keyboard controls
  - GUI interface with real-time plotting and interactive controls
- Comprehensive configuration system through `configs/config.py`
- Training and evaluation scripts with:
  - Tensorboard logging support
  - Early stopping mechanism
  - Model checkpointing
  - Performance visualization
- Batch files for Windows users for easy operation
- Complete documentation:
  - Detailed README.md with installation and usage instructions
  - Comprehensive user manual in ReStructuredText format
  - Contributing guidelines
- GitHub Actions CI/CD pipeline for automated testing
- Cross-platform support (Windows, Linux, macOS)
- MIT License

### Features
- **Environment**: Physics-based RF cavity simulation
- **Algorithm**: PPO (Proximal Policy Optimization) with optimized hyperparameters
- **Real-time Control**: Both command line and GUI interfaces
- **Monitoring**: Tensorboard integration for training visualization
- **Configuration**: Highly configurable through centralized config file
- **Platforms**: Windows, Linux, and macOS support
- **Python**: Compatible with Python 3.8+

### Dependencies
- gymnasium >= 0.29.0
- stable-baselines3 >= 2.0.0
- torch >= 1.13.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- scipy >= 1.7.0
- tensorboard >= 2.8.0

### Known Issues
- LLRF libraries need to be installed separately (platform-specific)
- Large memory usage during training with default settings
- GUI interface may require additional setup on some Linux distributions

### Performance
- Training typically achieves < 1 kHz mean absolute frequency detuning
- Optimized for CPU training (recommended over GPU for this use case)
- Supports parallel environments for faster training

## [Unreleased]

### Planned Features
- [ ] Noise models for more realistic simulation
- [ ] Multi-objective optimization (stability + efficiency)
- [ ] Support for different cavity configurations
- [ ] Transfer learning between different cavities
- [ ] Web-based monitoring dashboard
- [ ] Advanced control algorithms comparison (PID, LQR, MPC)
- [ ] Distributed control for multiple cavities
- [ ] Fault detection and diagnosis capabilities

---

For detailed information about each version, see the [releases page](https://github.com/iuming/SRF_FC_RL/releases).
