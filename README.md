# Fruit & Vegetable Classifier

This project is a machine learning application for classifying fruits and vegetables based on images. It uses a convolutional neural network (CNN) built with TensorFlow and Keras to identify different types of fruits and vegetables from an image dataset.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.7 or higher
- [Git](https://git-scm.com/)

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/nicolasbaldoino/fruit-vegetable-classifier.git
   ```

2. Navigate to the project directory:

   ```bash
   cd fruit-vegetable-classifier
   ```

3. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

4. Activate the virtual environment:

   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On Linux/MacOS:
     ```bash
     source venv/bin/activate
     ```

5. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

After installing the dependencies, you can run the main script:

```bash
python main.py
```

The model will be trained using the fruit and vegetable image dataset and will output performance metrics such as accuracy, confusion matrix, and classification report.

## Project Structure

```
fruit-vegetable-classifier/
│
├── dataset/                      # Data directory
├── data_preprocessing.py         # Data preprocessing script
├── evaluation.py                 # Script for model evaluation
├── main.py                       # Main script
├── model.py                      # Model definition
├── requirements.txt              # Required libraries
├── README.md                     # Project documentation
├── training.py                   # Script for model training
└── utils.py                      # Utility functions
```

## Dependencies

- **TensorFlow**: Used to build and train the neural network model.
- **Pandas**: Utilized for data manipulation.
- **NumPy**: Library for mathematical operations and array manipulation.
- **Seaborn**: Used for more attractive and informative data visualizations.
- **Matplotlib**: Plotting library used for visualizations.
- **Scikit-learn**: Used for model evaluation, including classification reports and confusion matrices.
- **Termcolor**: Adds color to terminal outputs.
- **Silence-TensorFlow**: Used to suppress TensorFlow warnings.

For the full list of dependencies, refer to the [requirements.txt](requirements.txt) file.

## Contributing

Contributions are welcome! Feel free to open issues and pull requests for code improvements, documentation, or new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
