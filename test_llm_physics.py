#!/usr/bin/env python3
import json
import pytest
import os
import sys
from datetime import datetime
import pandas as pd
from typing import Dict, List, Any, Optional
from pytest import main as pytest_main
import itertools


from llm_client import get_response, get_all_responses
from validators import validate_answer, DEBUG

# Enable debug mode if needed
# Set validators.DEBUG = True to enable debug print statements

# Constants
RESULTS_DIR = "results"
DEFAULT_MODELS = ["gpt", "deepseek", "perplexity"]
DATE_FORMAT = "%Y-%m-%d_%H-%M-%S"
DEBUG

# Store all results for summary
ALL_RESULTS = []

@pytest.fixture(scope="session")
def run_timestamp():
    return datetime.now().strftime(DATE_FORMAT)

def load_test_cases(path: str = "test_cases.json") -> List[Dict[str, Any]]:
    """
    Load test cases from JSON file.
    
    Args:
        path: Path to the test cases JSON file
        
    Returns:
        List of test case dictionaries
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_results_dir():
    """Create results directory if it doesn't exist"""
    os.makedirs(RESULTS_DIR, exist_ok=True)

def save_result(test_case: Dict[str, Any], model: str, validation_result: Dict[str, Any], 
                timestamp: str = None) -> None:
    """
    Save test result to CSV file and add to global results.
    
    Args:
        test_case: Test case dictionary
        model: Model name
        validation_result: Results from validator
        timestamp: Optional timestamp string (default: generate new)
    """
    ensure_results_dir()
    
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime(DATE_FORMAT)
    
    # Create results file if it doesn't exist
    results_file = os.path.join(RESULTS_DIR, f"results_{timestamp}.csv")
    
    # Create a unique test ID to avoid confusion with duplicate IDs across topics
    unique_test_id = f"{test_case['id']}_{test_case['topic']}"
    
    # Ensure actual_answer is a string
    actual_answer = str(validation_result.get("actual", "No response"))
    
    # Truncate actual answer for CSV display
    if len(actual_answer) > 100:
        actual_answer_display = actual_answer[:100] + "..."
    else:
        actual_answer_display = actual_answer
    
    # Format result for CSV
    result_row = {
        "test_id": unique_test_id,
        "id": test_case["id"],
        "topic": test_case["topic"],
        "question": test_case["question"][:100] + "..." if len(test_case["question"]) > 100 else test_case["question"],
        "model": model,
        "expected_answer": test_case["answer"],
        "actual_answer": actual_answer_display,
        "valid": validation_result.get("valid", False),
        "validation_method": validation_result.get("method", "unknown"),
        "tags": str(test_case.get("tags", {})),
        "timestamp": timestamp
    }
    
    # Add to global results for summary
    ALL_RESULTS.append({
        "test_id": unique_test_id,
        "id": test_case["id"],
        "topic": test_case["topic"],
        "question": test_case["question"],
        "model": model,
        "expected_answer": test_case["answer"],
        "actual_answer": actual_answer,
        "valid": validation_result.get("valid", False),
        "validation_method": validation_result.get("method", "unknown"),
        "validation_details": validation_result,
        "timestamp": timestamp
    })
    
    # Append to CSV
    df = pd.DataFrame([result_row])
    if os.path.exists(results_file):
        # Append without header
        df.to_csv(results_file, mode='a', header=False, index=False)
    else:
        # Create new file with header
        df.to_csv(results_file, mode='w', header=True, index=False)
    
    return results_file

def generate_report(timestamp: str, results_file: str = None) -> Optional[str]:
    """
    Generate summary report from test results.
    
    Args:
        timestamp: Timestamp string
        results_file: Optional results file path
        
    Returns:
        Path to the generated report file, or None if no results found
    """
    if results_file is None:
        results_file = os.path.join(RESULTS_DIR, f"results_{timestamp}.csv")
    
    # Check if we have results to process
    have_results = len(ALL_RESULTS) > 0
    file_exists = os.path.exists(results_file) and os.path.getsize(results_file) > 0
    
    if not have_results and not file_exists:
        print(f"No results found for timestamp: {timestamp}")
        return None
    
    # Use global results if available, otherwise load from file
    if have_results:
        results = ALL_RESULTS
    else:
        # Load results from CSV for report generation
        try:
            results_df = pd.read_csv(results_file)
            results = results_df.to_dict('records')
        except Exception as e:
            print(f"Error loading results from {results_file}: {e}")
            return None
    
    # Generate overall statistics
    total_tests = len(results)
    if total_tests == 0:
        print("No test results found")
        return None
    
    passed_tests = sum(1 for r in results if r["valid"])
    pass_rate = (passed_tests / total_tests) * 100
    
    # Model performance
    model_stats = {}
    for model in set(r["model"] for r in results):
        model_results = [r for r in results if r["model"] == model]
        model_tests = len(model_results)
        model_passed = sum(1 for r in model_results if r["valid"])
        model_rate = (model_passed / model_tests) * 100 if model_tests > 0 else 0
        model_stats[model] = {
            "tests": model_tests,
            "passed": model_passed,
            "pass_rate": model_rate
        }
    
    # Topic performance
    topic_stats = {}
    for topic in set(r["topic"] for r in results):
        topic_results = [r for r in results if r["topic"] == topic]
        topic_tests = len(topic_results)
        topic_passed = sum(1 for r in topic_results if r["valid"])
        topic_rate = (topic_passed / topic_tests) * 100 if topic_tests > 0 else 0
        topic_stats[topic] = {
            "tests": topic_tests,
            "passed": topic_passed,
            "pass_rate": topic_rate
        }
    
    # Group results by test case (using unique test_id)
    test_results = {}
    for result in results:
        test_id = result["test_id"]
        if test_id not in test_results:
            test_results[test_id] = {
                "id": result["id"],
                "topic": result["topic"],
                "question": result["question"],
                "expected": result["expected_answer"],
                "models": {}
            }
        
        # Store the result by model
        test_results[test_id]["models"][result["model"]] = {
            "valid": result["valid"],
            "actual": result["actual_answer"],
            "method": result.get("validation_method", "unknown")
        }
    
    # Build detailed results organized by test case and model
    detailed_results = {}
    for test_id, test_info in test_results.items():
        detailed_results[test_id] = {
            "id": test_info["id"],
            "topic": test_info["topic"],
            "question": test_info["question"],
            "expected": test_info["expected"],
            "models": test_info["models"]
        }
    
    # Build report
    report = {
        "timestamp": timestamp,
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "pass_rate": pass_rate,
        "model_performance": model_stats,
        "topic_performance": topic_stats,
        "detailed_results": detailed_results
    }
    
    # Save report as JSON
    ensure_results_dir()
    report_file = os.path.join(RESULTS_DIR, f"report_{timestamp}.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n===== PHYSICS LLM TEST REPORT =====")
    print(f"Time: {timestamp}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests} ({pass_rate:.2f}%)")
    print("\nModel Performance:")
    for model, stats in model_stats.items():
        print(f"  {model}: {stats['passed']}/{stats['tests']} ({stats['pass_rate']:.2f}%)")
    print("\nTopic Performance:")
    for topic, stats in topic_stats.items():
        print(f"  {topic}: {stats['passed']}/{stats['tests']} ({stats['pass_rate']:.2f}%)")
    
    # Print detailed results table
    print("\nDetailed Results:")
    for test_id, test_info in detailed_results.items():
        print(f"\nTest {test_info['id']} ({test_info['topic']}):")
        print(f"  Question: {test_info['question'][:80]}..." if len(test_info['question']) > 80 else f"  Question: {test_info['question']}")
        print(f"  Expected: {test_info['expected']}")
        
        for model, model_result in test_info['models'].items():
            status = "✓" if model_result['valid'] else "✗"
            answer = model_result['actual']
            if isinstance(answer, str) and len(answer) > 80:
                answer = answer[:77] + "..."
            print(f"  {model} [{status}]: {answer}")
    
    print(f"\nDetailed report saved to: {report_file}")
    print("==================================\n")
    
    return report_file

def run_single_test_case(tc: Dict[str, Any], model: str, timestamp: str) -> Dict[str, Any]:
    """
    Run a single test case with a specific model and return the result.
    
    Args:
        tc: Test case dictionary
        model: Model name
        timestamp: Timestamp string
        
    Returns:
        Validation result dictionary
    """
    # Extract parameter values
    parameters = tc.get("parameters", {})
    
    try:
        # Get model response
        response = get_response(
            question=tc["question"],
            model=model,
            numeric_only=parameters.get("numeric_only") == "True",
            multi_part=parameters.get("multi_part") == "True",
            image_url=parameters.get("image_url") if parameters.get("image_url") != "None" else None
        )
        
        # Validate response
        validation_result = validate_answer(tc["answer"], response, parameters)
        
    except Exception as e:
        # Create error result instead of failing
        error_message = str(e)
        print(f"Error running test case {tc['id']} with model {model}: {error_message}")
        validation_result = {
            "valid": False,
            "method": "error",
            "reason": f"API Error: {error_message}",
            "actual": f"ERROR: {error_message}"
        }
    
    # Save result - happens even for exceptions
    save_result(tc, model, validation_result, timestamp)
    
    return validation_result

class TestLLMPhysics:
    """Test class for LLM physics questions."""
    
    @pytest.fixture(scope="class")
    def timestamp(self):
        """Generate a timestamp for this test run."""
        return datetime.now().strftime(DATE_FORMAT)
    
    @pytest.fixture(scope="class")
    def test_cases(self):
        """Load all test cases."""
        return load_test_cases()

    # Test all questions with each model

@pytest.mark.parametrize("tc,model", itertools.product(load_test_cases(), DEFAULT_MODELS))
def test_question_model_pair(tc, model, run_timestamp):
    """Test a single question with a specific model."""
    try:
        validation_result = run_single_test_case(tc, model, run_timestamp)
        # We still include the assert for pytest reporting, but the result is already saved
        assert validation_result["valid"], \
            f"Test case {tc['id']} ({tc['topic']}) failed with model {model}"
    except Exception as e:
        # This should rarely happen now since run_single_test_case handles exceptions
        print(f"Unhandled exception in test_question_model_pair: {str(e)}")
        # Re-raise so pytest registers the failure
        pytest.fail(f"Error running test case {tc['id']} with model {model}: {str(e)}")



def run_tests(models: Optional[List[str]] = None, quiet: bool = False) -> Optional[str]:
    """
    Run all tests and generate a report.
    
    Args:
        models: List of models to test (default: all models)
        quiet: Suppress pytest output (default: False)
        
    Returns:
        Path to the report file, or None if report generation failed
    """
    # Reset global results
    global ALL_RESULTS
    ALL_RESULTS = []
    
    # Set models for testing
    if models is not None:
        # Update DEFAULT_MODELS for this run
        global DEFAULT_MODELS
        DEFAULT_MODELS = models
    
    timestamp = datetime.now().strftime(DATE_FORMAT)
    print(f"Starting physics LLM tests with timestamp: {timestamp}")
    print(f"Testing models: {DEFAULT_MODELS}")
    
    # Make sure results directory exists
    ensure_results_dir()
    
    # Load test cases
    try:
        test_cases = load_test_cases()
    except Exception as e:
        print(f"Error loading test cases: {e}")
        return None
    
    if not test_cases:
        print("No test cases found.")
        return None
    
    # Dictionary to track test cases by ID to avoid duplicates 
    # in summary (some test cases share IDs across topics)
    processed_tests = {}
    
    # Run tests manually to avoid pytest output
    if quiet:
        for tc_index, tc in enumerate(test_cases):
            # Generate a unique test identifier in case of duplicate IDs
            unique_id = f"{tc['id']}_{tc['topic']}"
            processed_tests[unique_id] = {
                "id": tc['id'],
                "topic": tc['topic'],
                "index": tc_index
            }
            
            for model in DEFAULT_MODELS:
                try:
                    run_single_test_case(tc, model, timestamp)
                    print(f"Test case {tc['id']} ({tc['topic']}) with model {model}: ", end="")
                    if ALL_RESULTS[-1]["valid"]:
                        print("✓ PASS")
                    else:
                        print("✗ FAIL")
                except Exception as e:
                    print(f"Error running test case {tc['id']} with model {model}: {str(e)}")
    else:
        # Build pytest args
        pytest_args = ["-v", __file__]
        
        # Run pytest
        result = pytest_main(pytest_args)
    
    # Generate report
    report_file = generate_report(timestamp)
    
    return report_file

def print_summary_from_report(report_file: str) -> None:
    """
    Print a clean summary from a report file.
    
    Args:
        report_file: Path to the report JSON file
    """
    try:
        with open(report_file, 'r') as f:
            report = json.load(f)
        
        print("\n===== SUMMARY =====")
        print(f"Overall: {report['passed_tests']}/{report['total_tests']} tests passed ({report['pass_rate']:.2f}%)")
        
        print("\nTest cases:")
        for test_id, test_info in report['detailed_results'].items():
            print(f"\nTest {test_info['id']}: {test_info['topic']}")
            print(f"Expected: {test_info['expected']}")
            
            # Print all model results for this test
            for model, result in test_info['models'].items():
                status = "✓" if result['valid'] else "✗"
                actual = result['actual']
                if isinstance(actual, str) and len(actual) > 80:
                    actual = actual[:77] + "..."
                print(f"{model} [{status}]: {actual}")
    except FileNotFoundError:
        print(f"Report file not found: {report_file}")
    except json.JSONDecodeError:
        print(f"Error parsing report file: {report_file}")
    except Exception as e:
        print(f"Error reading report: {e}")

if __name__ == "__main__":
    # Handle command line args
    models = DEFAULT_MODELS
    quiet = False
    
    # Check for command line arguments
    for arg in sys.argv[1:]:
        if arg == "--quiet" or arg == "-q":
            quiet = True
        elif arg.startswith("-"):
            print(f"Unknown option: {arg}")
        else:
            # Collect model names
            if models == DEFAULT_MODELS:
                models = [arg]
            else:
                models.append(arg)
    
    # Run tests
    report_file = run_tests(models, quiet)
    
    # Print clean summary
    if report_file:
        print_summary_from_report(report_file)
    else:
        print("No test report was generated. Please check the logs for errors.") 