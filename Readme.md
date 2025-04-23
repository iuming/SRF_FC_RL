![# SRF FC RL](assets/logo.png)
[![GitHub Repo stars](https://img.shields.io/github/stars/iuming/SRF_FC_RL?style=social)](https://github.com/iuming/SRF_FC_RL/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/iuming/SRF_FC_RL)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/iuming/SRF_FC_RL)](https://github.com/iuming/SRF_FC_RL/commits/master)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/iuming/SRF_FC_RL/pulls)

# Superconducting RadioFrequency cavity Frequency Control by Reinforcement Learning

Control the frequency of the particle accelerator's radio frequency superconducting cavity based on reinforcement learning algorithm.


## Project Structure

Describe the directory structure of the project to help users understand the files and folders included in the project.

```
SRF_FC_RL/
├── config/       # Configuration files
├── envs/         # Environment implementations
├── models/       # Trained models
├── scripts/      # Training/Evaluation scripts
├── utils/        # Utility modules
├── README.md     # Project documentation
└── requirements.txt # Dependency list
```

## Installation Guide

- Install conda 
- (Suggested) Create a virtual environment and activate it
- Install dependencies

1. Create a virtual environment:
```bash
conda create -n RL python==3.13.2
conda activate RL
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Train the Model
```bash
python scripts/train.py --config config/config.yaml
```

## Evaluate the Model
```bash
python scripts/evaluate.py --config config/config.yaml
```

## Notes
1. Ensure that the CUDA environment is installed to enable GPU acceleration.
2. Parameters in the configuration file can be adjusted based on actual requirements.
3. Visualization results will automatically pop up after evaluation is completed.

## Contributing

If you want others to contribute to the project, provide contribution guidelines and contact information.

## License

This repository is licensed under the [LICENSE](LICENSE).

## Contact

Provide contact information so users can reach out to you.

Feel free to customize this template to fit the specific needs of your project. If you have any specific requirements or need further assistance, please let me know. I'd be happy to help!

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=iuming/SRF_FC_RL&type=Date)](https://star-history.com/#iuming/SRF_FC_RL&Date)

