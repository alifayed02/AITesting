# Physics LLM Testing Framework

A framework for automatically testing and evaluating LLM responses to physics questions.

## Overview

This project provides tools to:
1. Submit physics questions to multiple LLMs (GPT, DeepSeek, Perplexity)
2. Validate responses against expected answers
3. Generate performance reports

## Files

- `llm_client.py` - Interface for interacting with different LLM providers
- `validators.py` - Functions to validate LLM responses against expected answers
- `test_llm_physics.py` - Pytest test suite for running tests
- `test_cases.json` - Physics test cases with questions, expected answers, and parameters

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up API keys as environment variables:
```bash
export OPENAI_API_KEY="your-openai-key"
export DEEPSEEK_API_KEY="your-deepseek-key"
export PERPLEXITY_API_KEY="your-perplexity-key"
```

## Running Tests

Run all tests with all models:
```bash
python test_llm_physics.py
```

Run tests with specific models:
```bash
python test_llm_physics.py gpt deepseek
```

Run tests using pytest directly:
```bash
pytest -v test_llm_physics.py
```

## Test Case Format

Test cases in `test_cases.json` follow this format:
```json
{
  "topic": "Thermodynamics",
  "id": "1",
  "question": "The physics question...",
  "answer": "29.2°C",
  "parameters": {
    "numeric_only": "True",
    "multi_part": "False",
    "image_url": "None"
  },
  "tags": {
    "category": "Question Type",
    "subcategory": "Mathematical",
    "complexity": "Intermediate",
    "application": "Engineering",
    "format": "Calculation"
  }
}
```

## Debugging

To enable detailed debug output, set the `DEBUG` flag in `validators.py`:
```python
DEBUG = True  # Set to False to disable debug prints
```

## Results

Test results are saved to:
- `results/results_TIMESTAMP.csv` - Detailed test results
- `results/report_TIMESTAMP.json` - Summary report with statistics

The report shows overall performance and breakdowns by model and topic. 