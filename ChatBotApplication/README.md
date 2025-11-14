# Persona Comparison Analysis

This project contains data and analysis for comparing different chatbot personas. It evaluates how different personas respond to a set of questions and measures the similarity and quality of their answers using various metrics.

The comparisons are stored in `comparisons/persona_comparisons.csv` and include the following metrics:
- `cosine_sim`: Cosine similarity between persona responses.
- `rouge_l_f1`: ROUGE-L F1 score to measure content overlap.
- `gpt_score_0_100`: A GPT-based evaluation score on a scale of 0 to 100.

## Prerequisites

- Python 3.9 or higher

## Setup

Follow these steps to set up your local development environment.

### 1. Create and Activate a Virtual Environment

A virtual environment helps to manage project dependencies and avoid conflicts with other projects.

**On macOS and Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**

```bash
python -m venv venv
.\venv\Scripts\activate
```

Your command prompt should now be prefixed with `(venv)`, indicating that the virtual environment is active.

### 2. Install Dependencies

This project's dependencies are listed in `requirements.txt`. To install them, run the following command in your activated virtual environment:

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` file, you can create one with the following content, based on the libraries used in the project context:

```
# requirements.txt
pandas
numpy
langchain
langchain-community
langchain-openai
argilla
setuptools
```

## Usage

The core data of this project is the `comparisons/persona_comparisons.csv` file. You can use this data for analysis, visualization, or further processing.

### Running the Analysis

While the CSV file itself is not executable, you can use a Python script to read and analyze its contents. Here is an example of how to load and display the data using the `pandas` library:

Create a Python file (e.g., `analyze.py`) with the following code:

```python
import pandas as pd

def analyze_persona_comparisons(file_path='comparisons/persona_comparisons.csv'):
    """
    Loads and displays the persona comparison data.
    """
    try:
        df = pd.read_csv(file_path)
        print("Persona Comparison Analysis:")
        print(df.head())
        print("\nBasic Statistics:")
        print(df.describe())
    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")

if __name__ == '__main__':
    analyze_persona_comparisons()
```

To run this script, execute the following command in your terminal:

```bash
python analyze.py
```

### Expected Output

Running the `analyze.py` script will print the first few rows of the `persona_comparisons.csv` file to your console, followed by a statistical summary of the data, giving you a quick overview of the comparison metrics.