# Contributing to RF Cavity Control System

Thank you for your interest in contributing to the RF Cavity Control System! This document provides guidelines for contributing to this project.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Knowledge of reinforcement learning (helpful but not required)
- Understanding of RF cavity physics (helpful for advanced contributions)

### Development Setup

1. **Fork the repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/ML_Learning.git
   cd ML_Learning/RL_Learning/custom/20250725
   ```

2. **Set up development environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

3. **Verify installation**
   ```bash
   python main.py env-test
   ```

## How to Contribute

### Types of Contributions

We welcome several types of contributions:

1. **Bug Reports**: Found a bug? Please report it!
2. **Feature Requests**: Have an idea for improvement?
3. **Code Contributions**: Bug fixes, new features, optimizations
4. **Documentation**: Improve docs, add examples, write tutorials
5. **Testing**: Add tests, improve test coverage

### Reporting Bugs

When reporting bugs, please include:

- **Clear description** of the problem
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, package versions)
- **Code snippets** or error messages
- **Possible solutions** if you have ideas

**Bug Report Template:**
```markdown
## Bug Description
[Clear description of the bug]

## Steps to Reproduce
1. 
2. 
3. 

## Expected Behavior
[What should happen]

## Actual Behavior
[What actually happens]

## Environment
- OS: [e.g., Windows 10, Ubuntu 20.04]
- Python: [e.g., 3.9.7]
- Package versions: [pip freeze output]

## Additional Context
[Any other relevant information]
```

### Feature Requests

For feature requests, please provide:

- **Clear description** of the proposed feature
- **Use case**: Why would this be useful?
- **Proposed implementation** (if you have ideas)
- **Alternatives considered**

### Code Contributions

#### Before You Start

1. **Check existing issues** to see if someone is already working on it
2. **Create an issue** to discuss major changes before implementing
3. **Start small** for your first contribution

#### Development Workflow

1. **Create a branch** for your feature/fix
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-number
   ```

2. **Make your changes**
   - Follow the coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   # Run basic tests
   python main.py env-test
   python -m pytest tests/  # If tests exist
   
   # Test the main functionality
   python main.py train  # Quick training test
   python main.py test   # Testing with existing model
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```

5. **Push and create pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

#### Coding Standards

- **Python Style**: Follow PEP 8
- **Docstrings**: Use Google-style docstrings
- **Type Hints**: Add type hints for function parameters and returns
- **Comments**: Explain complex logic, not obvious code
- **Variable Names**: Use descriptive names

**Example:**
```python
def calculate_reward(self, observation: np.ndarray) -> float:
    """
    Calculate reward based on frequency detuning.
    
    Args:
        observation: Current system observation containing detuning info
        
    Returns:
        Reward value (negative absolute detuning)
    """
    detuning = observation[3]  # Frequency detuning in kHz
    return -np.abs(detuning)
```

#### Documentation Standards

- **Header Comments**: All files should have comprehensive headers
- **Function Documentation**: Document all public functions
- **README Updates**: Update README for new features
- **Examples**: Provide usage examples for new features

#### Testing Guidelines

- **Test Coverage**: Aim for good test coverage of new code
- **Unit Tests**: Test individual functions/methods
- **Integration Tests**: Test component interactions
- **Performance Tests**: For performance-critical changes

### Pull Request Process

1. **Ensure your PR**:
   - Has a clear title and description
   - References related issues
   - Includes tests for new functionality
   - Updates documentation as needed
   - Passes all existing tests

2. **PR Description Template**:
   ```markdown
   ## Changes Made
   [Brief description of changes]
   
   ## Related Issues
   Closes #[issue number]
   
   ## Testing
   - [ ] Tested locally
   - [ ] Added new tests
   - [ ] All tests pass
   
   ## Documentation
   - [ ] Updated README if needed
   - [ ] Added docstrings
   - [ ] Updated comments
   
   ## Additional Notes
   [Any other relevant information]
   ```

3. **Review Process**:
   - Maintainer will review your PR
   - Address any feedback promptly
   - Be open to suggestions and changes
   - Once approved, your PR will be merged

## Project Structure

Understanding the project structure helps with contributions:

```
RL_Learning/custom/20250725/
├── src/                    # Core source code
│   ├── rf_cavity_env.py   # Main environment
│   └── realtime_control.py # Real-time control wrapper
├── scripts/               # Executable scripts
│   ├── train_rf_cavity.py # Training script
│   ├── test_rf_cavity.py  # Testing script
│   └── realtime_*.py      # Real-time interfaces
├── configs/               # Configuration files
├── tests/                 # Test files (to be added)
├── docs/                  # Documentation (to be added)
└── examples/              # Usage examples (to be added)
```

## Specific Contribution Areas

### High Priority
1. **Test Suite**: Add comprehensive tests
2. **Documentation**: Improve inline documentation
3. **Examples**: Create more usage examples
4. **Performance**: Optimize training and simulation

### Medium Priority
1. **New Features**: Implement items from Future Improvements
2. **GUI Improvements**: Enhance real-time interface
3. **Configuration**: Make system more configurable
4. **Error Handling**: Improve error messages and recovery

### Advanced Contributions
1. **Physics Models**: Improve RF cavity simulation
2. **RL Algorithms**: Add support for other algorithms
3. **Multi-cavity**: Support for multiple cavity control
4. **Web Interface**: Web-based monitoring dashboard

## Community Guidelines

- **Be respectful**: Treat all contributors with respect
- **Be patient**: Everyone is learning and contributing their time
- **Be helpful**: Help others when you can
- **Ask questions**: Don't hesitate to ask if you're unsure
- **Have fun**: Enjoy the learning and contribution process!

## Recognition

Contributors will be:
- Listed in the project README
- Mentioned in release notes for significant contributions
- Invited to become maintainers for substantial ongoing contributions

## Getting Help

If you need help:

1. **Check documentation**: README, code comments, docstrings
2. **Search issues**: Someone might have asked the same question
3. **Create an issue**: For questions or problems
4. **Contact maintainer**: ming.liu@example.com

## Code of Conduct

This project follows a simple code of conduct:

- **Be respectful** to all participants
- **Be inclusive** and welcoming to newcomers
- **Focus on the work** and keep discussions professional
- **Help create a positive** learning and development environment

Thank you for contributing to the RF Cavity Control System! 🚀
