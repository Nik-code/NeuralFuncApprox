# NeuralFuncApprox: Visualizing Neural Network Function Approximation

NeuralFuncApprox is an educational project that demonstrates how neural networks learn to approximate mathematical functions. Through animated visualizations, it showcases the learning process of a neural network as it attempts to model various mathematical functions.

## Project Overview

This project uses PyTorch to implement a simple neural network and matplotlib to create animated visualizations of the learning process. The neural network attempts to learn different mathematical functions, and the progress is displayed in real-time, allowing viewers to see how the network's predictions improve over time.

## Features

- Implementation of a basic neural network using PyTorch
- Real-time visualization of the learning process
- Support for multiple mathematical functions
- Animated GIFs showcasing the learning progression

## Examples

Here are some examples of the neural network learning different functions:

1. Cubic Function (x³ - 3x)

![Cubic Function Learning](Gifs/cubic%20function.gif)

2. Quadratic Function (x² - 2x + 1)

![Quadratic Function Learning](Gifs/quad%20function.gif)

3. Sine Function (sin(2πx))

![Sine Function Learning](Gifs/sin%20function.gif)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/NeuralFuncApprox.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main script to generate the animations:

```
python neural_func_approx.py
```

You can modify the `actual_function` in the script to experiment with different mathematical functions.

## Requirements

This project requires Python 3.x and the following libraries:

- numpy
- matplotlib
- torch

For specific version requirements, please refer to the [requirements.txt](requirements.txt) file.

## Contributing

Contributions to improve the project or add new features are welcome! Please feel free to submit a pull request or open an issue to discuss potential changes/additions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch team for their excellent deep learning framework
- Matplotlib developers for the powerful visualization tools
