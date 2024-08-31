import warnings

from silence_tensorflow import silence_tensorflow

# A function to avoid tensorflow warnings
silence_tensorflow()
warnings.filterwarnings('ignore')
