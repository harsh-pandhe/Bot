# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-04

### Added
- Initial public release
- `bot_ultimate.ino` - Main robot firmware with neural network integration
- Comprehensive neural network training pipeline (`train_comprehensive.py`)
- Multi-sensor fusion (front, left, right proximity sensors)
- Encoder-based positioning and turning
- OLED display support for real-time debugging
- Data collection utilities for training dataset generation
- Safety override system to prevent crashes
- Hybrid control mode (neural network + rule-based)

### Features
- Neural network inference in < 1ms on Teensy
- 12-bit ADC sensor readings (0-4095 range)
- PWM-based motor control (0-255)
- Precision turns using encoder feedback
- Multi-layer safety checks
- Real-time status display

### Documentation
- Comprehensive README with quick start guide
- Hardware requirements and pinout documentation
- Configuration parameters explained
- Neural network architecture details
- Data format specifications

## [0.9.0] - Development Version

### Added
- Multiple training script variants for experimentation
- Data collection firmware versions
- Calibration utilities
- Extensive training datasets

### Notes
- Project was in active development phase
- Multiple algorithm iterations tested
- Various sensor configurations evaluated

---

## Future Roadmap

### [1.1.0] - Planned
- [ ] Vision-based navigation support
- [ ] Path planning optimization
- [ ] Real-time learning/adaptation
- [ ] Multi-maze generalization
- [ ] Advanced PID tuning

### [1.2.0] - Research Phase
- [ ] ROS integration
- [ ] Simulated environment (Gazebo)
- [ ] Advanced ML models (CNN for vision)
- [ ] Cloud connectivity for data analysis

---

**Current Status**: Version 1.0.0 - Stable Release 🎉

For detailed commit history, see git log.
