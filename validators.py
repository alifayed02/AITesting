import re
import json
from typing import Dict, Any, Optional, Union
from difflib import SequenceMatcher
import math
from collections import Counter

# Debug mode flag - set to True to enable print statements
DEBUG = True

def debug_print(*args, **kwargs):
    """Print only if DEBUG mode is enabled"""
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)

def validate_answer(expected: str, actual: str, parameters: Dict[str, str]) -> Dict[str, Any]:
    """
    Validate LLM responses against expected answers based on response type.
    
    Args:
        expected: The expected answer string
        actual: The actual response from the LLM
        parameters: Parameters from the test case that determine validation approach
        
    Returns:
        Dict with validation results including valid flag and details
    """
    debug_print(f"Validating answer with parameters: {parameters}")
    debug_print(f"Expected answer: {expected}")
    debug_print(f"Actual answer: {actual[:100]}..." if len(str(actual)) > 100 else f"Actual answer: {actual}")
    
    # Determine validation strategy based on parameters
    numeric_only = parameters.get('numeric_only') == 'True'
    multi_part = parameters.get('multi_part') == 'True'
    
    if multi_part:
        debug_print("Using multi-part validation")
        result = validate_multi_part(expected, actual)
    elif numeric_only:
        debug_print("Using numeric validation")
        result = validate_numeric(expected, actual)
    else:
        debug_print("Using text validation")
        result = validate_text(expected, actual)
    
    # Add the actual answer to the result for reference
    if 'actual' not in result:
        result['actual'] = actual
    
    debug_print(f"Validation result: valid={result['valid']}, method={result.get('method', 'N/A')}")
    return result

def validate_numeric(expected: str, actual: str) -> Dict[str, Any]:
    """
    Validate numeric answers, allowing for minor variations in formatting.
    
    Args:
        expected: Expected numeric answer (potentially with units)
        actual: Actual response from LLM
    
    Returns:
        Dict with validation results
    """
    debug_print(f"Numeric validation - expected: {expected}, actual: {actual}")
    
    # Try to extract the last numeric value from a long answer
    if len(actual) > 100:
        # Look for a numeric value near the end of the response
        # This helps with models that show their work and then give the answer
        last_paragraph = actual.split('\n\n')[-1].strip()
        debug_print(f"Checking last paragraph for numeric answer: {last_paragraph}")
        
        # Try to find a number at the end of the text
        number_match = re.search(r'(?:^|\s)(\d+\.?\d*|\d+/\d+)\s*([a-zA-Z°%]*)(?:$|\s|\.)', last_paragraph)
        if number_match:
            actual = number_match.group(0).strip()
            debug_print(f"Found numeric value at end: {actual}")
            
        # Also try looking for bolded answers like "**28°C**"
        bold_match = re.search(r'\*\*([\d\.\s\/]+\s*[a-zA-Z°%]*)\*\*', last_paragraph)
        if bold_match:
            actual = bold_match.group(1).strip()
            debug_print(f"Found bolded numeric value: {actual}")
        
        # One more try - look for statements like "approximately 3.80" or "is 3.80"
        approx_match = re.search(r'(?:is|approximately|about|around|equals|=)\s*([\d\.\s\/]+\s*[a-zA-Z°%]*)', last_paragraph, re.IGNORECASE)
        if approx_match:
            actual = approx_match.group(1).strip()
            debug_print(f"Found approximate value statement: {actual}")
    
    # Handle fractions in expected or actual
    expected_fraction_match = re.search(r'(\d+)/(\d+)', expected)
    actual_fraction_match = re.search(r'(\d+)/(\d+)', actual)
    
    # If either is a fraction, convert both to decimal for comparison
    if expected_fraction_match or actual_fraction_match:
        debug_print("Fraction detected, converting to decimal for comparison")
        
        if expected_fraction_match:
            num, denom = expected_fraction_match.groups()
            expected_decimal = float(num) / float(denom)
            debug_print(f"Converted expected fraction {expected} to {expected_decimal}")
            expected = str(expected_decimal)
        
        if actual_fraction_match:
            num, denom = actual_fraction_match.groups()
            actual_decimal = float(num) / float(denom)
            debug_print(f"Converted actual fraction {actual} to {actual_decimal}")
            actual = str(actual_decimal)
    
    # Extract numbers from both strings
    expected_val, expected_unit = extract_number_and_unit(expected)
    actual_val, actual_unit = extract_number_and_unit(actual)
    
    debug_print(f"Extracted - expected: {expected_val} {expected_unit}, actual: {actual_val} {actual_unit}")
    
    if expected_val is None or actual_val is None:
        debug_print("Could not extract numeric values")
        return {
            "valid": False,
            "reason": "Could not extract numeric values",
            "expected": expected,
            "actual": actual
        }
    
    # Check numeric equality (with tolerance)
    # Use a relative tolerance for larger numbers and absolute for smaller
    abs_tolerance = 0.01
    rel_tolerance = 0.05  # 5% relative tolerance
    
    # Determine which tolerance to use based on the expected value
    if abs(expected_val) > 1.0:
        # Use relative tolerance for larger numbers
        tolerance = abs(expected_val) * rel_tolerance
        numeric_match = abs(expected_val - actual_val) <= tolerance
        debug_print(f"Using relative tolerance: {tolerance}")
    else:
        # Use absolute tolerance for smaller numbers
        tolerance = abs_tolerance
        numeric_match = abs(expected_val - actual_val) <= tolerance
        debug_print(f"Using absolute tolerance: {tolerance}")
    
    # Check units if present in expected
    unit_match = True
    if expected_unit and actual_unit:
        unit_match = expected_unit.lower() == actual_unit.lower()
        if not unit_match:
            debug_print(f"Unit mismatch: expected '{expected_unit}', got '{actual_unit}'")
    
    debug_print(f"Numeric match: {numeric_match}, Unit match: {unit_match}")
    
    return {
        "valid": numeric_match and unit_match,
        "numeric_match": numeric_match,
        "unit_match": unit_match,
        "expected_value": expected_val,
        "expected_unit": expected_unit,
        "actual_value": actual_val,
        "actual_unit": actual_unit,
        "difference": abs(expected_val - actual_val),
        "tolerance": tolerance,
        "actual": actual
    }

def validate_multi_part(expected: str, actual: str) -> Dict[str, Any]:
    """
    Validate multi-part answers, typically in format "a. X; b. Y; c. Z"
    
    Args:
        expected: Expected multi-part answer
        actual: Actual response from LLM (should be JSON)
    
    Returns:
        Dict with validation results
    """
    debug_print(f"Multi-part validation - expected: {expected}")
    debug_print(f"Actual type: {type(actual)}")
    
    # Check if the actual response contains letters like a, b, c with values
    if isinstance(actual, str) and not actual.startswith('{'):
        # Try to parse a response like "(a) 100 T (b) 2 A (c) 0.50 H"
        parse_result = {}
        
        # Look for patterns like (a) value, (b) value, etc.
        part_matches = re.findall(r'(?:\()?([a-z])(?:\))?\s*[.:)]\s*([^(]+)', actual)
        if part_matches:
            for letter, value in part_matches:
                parse_result[letter] = value.strip()
            
            debug_print(f"Parsed non-JSON response into parts: {parse_result}")
            actual = parse_result
    
    # For multi-part answers, we expect actual to be a dict (from JSON)
    if not isinstance(actual, dict):
        try:
            # Try to parse it if it's a string containing JSON
            if isinstance(actual, str):
                debug_print("Attempting to parse JSON string")
                actual = json.loads(actual)
                debug_print(f"Parsed JSON: {actual}")
        except Exception as e:
            debug_print(f"JSON parsing failed: {e}")
            return {
                "valid": False,
                "reason": "Expected JSON object for multi-part answer",
                "expected": expected,
                "actual": actual
            }
    
    # Parse expected answer format "a. X; b. Y; c. Z"
    expected_parts = {}
    for part in expected.split(';'):
        part = part.strip()
        if not part:
            continue
        
        # Extract part letter and answer
        match = re.match(r'([a-z])[.)]\s*(.*)', part)
        if match:
            letter, answer = match.groups()
            expected_parts[letter] = answer.strip()
    
    debug_print(f"Parsed expected parts: {expected_parts}")
    
    if not expected_parts:
        debug_print("Could not parse expected multi-part answer")
        return {
            "valid": False,
            "reason": "Could not parse expected multi-part answer",
            "expected": expected,
            "actual": actual
        }
    
    # Compare each part
    part_results = {}
    all_valid = True
    
    for letter, expected_value in expected_parts.items():
        debug_print(f"Validating part {letter}")
        if letter in actual:
            actual_value = str(actual[letter])
            debug_print(f"Part {letter} - expected: {expected_value}, actual: {actual_value}")
            
            # Validate this part using appropriate strategy
            if re.search(r'^\s*[\d.]+\s*[a-zA-Z°%]*\s*$', expected_value) or '/' in expected_value:
                debug_print(f"Using numeric validation for part {letter}")
                result = validate_numeric(expected_value, actual_value)
            else:
                debug_print(f"Using text validation for part {letter}")
                result = validate_text(expected_value, actual_value)
            
            part_results[letter] = result
            all_valid = all_valid and result["valid"]
            debug_print(f"Part {letter} validation result: {result['valid']}")
        else:
            debug_print(f"Part {letter} missing from response")
            part_results[letter] = {
                "valid": False,
                "reason": f"Part {letter} missing from response"
            }
            all_valid = False
    
    debug_print(f"Overall multi-part validation result: {all_valid}")
    
    return {
        "valid": all_valid,
        "parts": part_results,
        "expected_parts": expected_parts,
        "actual_parts": actual,
        "actual": actual
    }

def validate_text(expected: str, actual: str) -> Dict[str, Any]:
    debug_print(f"Text validation - expected: {expected}")
    debug_print(f"Actual: {actual[:100]}..." if len(str(actual)) > 100 else f"Actual: {actual}")
    
    # Check for exact match
    if expected.strip() == actual.strip():
        debug_print("Exact match found")
        return {
            "valid": True,
            "method": "exact_match",
            "expected": expected,
            "actual": actual
        }
    
    # Normalize both texts (lowercase, remove punctuation)
    expected_norm = normalize_text(expected)
    actual_norm = normalize_text(actual)
    
    debug_print(f"Normalized expected: {expected_norm}")
    debug_print(f"Normalized actual: {actual_norm}")
    
    # Check for normalized match
    if expected_norm == actual_norm:
        debug_print("Normalized match found")
        return {
            "valid": True,
            "method": "normalized_match",
            "expected": expected,
            "actual": actual
        }
    
    # Calculate similarity using multiple metrics
    seq_similarity = SequenceMatcher(None, expected_norm, actual_norm).ratio()
    cosine_sim = cosine_similarity(expected_norm, actual_norm)
    
    debug_print(f"Sequence similarity: {seq_similarity}")
    debug_print(f"Cosine similarity: {cosine_sim}")
    
    # Use the best similarity score
    similarity = max(seq_similarity, cosine_sim)
    similarity_method = "cosine" if cosine_sim > seq_similarity else "sequence"
    
    # Threshold for similarity
    threshold = 0.7
    
    # Check if expected text is contained in actual
    contains_expected = expected_norm in actual_norm
    debug_print(f"Contains expected: {contains_expected}")
    
    # Validate against similarity or containment
    valid = similarity >= threshold or contains_expected
    
    method = "no_match"
    if similarity >= threshold:
        method = f"{similarity_method}_similarity"
    elif contains_expected:
        method = "containment"
    
    debug_print(f"Validation method: {method}, valid: {valid}")
    
    return {
        "valid": valid,
        "method": method,
        "sequence_similarity": seq_similarity,
        "cosine_similarity": cosine_sim, 
        "best_similarity": similarity,
        "threshold": threshold,
        "contains_expected": contains_expected,
        "expected": expected,
        "actual": actual
    }

def extract_number_and_unit(text: str) -> tuple:
    if not isinstance(text, str):
        debug_print(f"Non-string input to extract_number_and_unit: {type(text)}")
        return None, None
        
    # Match a number (including scientific notation) with optional units
    # Can capture numbers like: 123, 123.45, 1.23e-4, 1.23 × 10^-4
    pattern = r'(-?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*(?:×\s*10\^([-+]?\d+))?\s*([a-zA-Z°%]*)'
    match = re.search(pattern, text)
    
    if match:
        number_str, exponent, unit = match.groups()
        try:
            value = float(number_str)
            # Apply scientific notation if present
            if exponent:
                value *= 10 ** float(exponent)
            debug_print(f"Extracted number: {value}, unit: '{unit}'")
            return value, unit.strip()
        except ValueError as e:
            debug_print(f"Value conversion error: {e}")
            pass
    else:
        debug_print(f"No numeric pattern match in: '{text}'")
    
    return None, None

def normalize_text(text: str) -> str:
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation except for significant symbols
    text = re.sub(r'[.,;:!?"\'\(\)\[\]\{\}]', '', text)
    
    # Standardize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def cosine_similarity(text1: str, text2: str) -> float:
    """
    Calculate cosine similarity between two text strings.
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Cosine similarity score between 0 and 1
    """
    # Tokenize the texts into words
    words1 = text1.split()
    words2 = text2.split()
    
    debug_print(f"Cosine similarity calculation - words in text1: {len(words1)}, words in text2: {len(words2)}")
    
    # Create word frequency dictionaries
    word_counts1 = Counter(words1)
    word_counts2 = Counter(words2)
    
    # Create a unified set of words
    all_words = set(word_counts1.keys()) | set(word_counts2.keys())
    
    # Calculate dot product
    dot_product = sum(word_counts1.get(word, 0) * word_counts2.get(word, 0) for word in all_words)
    
    # Calculate magnitudes
    magnitude1 = math.sqrt(sum(count**2 for count in word_counts1.values()))
    magnitude2 = math.sqrt(sum(count**2 for count in word_counts2.values()))
    
    # Handle zero division
    if magnitude1 == 0 or magnitude2 == 0:
        debug_print("Zero magnitude detected in cosine similarity calculation")
        return 0.0
    
    similarity = dot_product / (magnitude1 * magnitude2)
    debug_print(f"Cosine similarity result: {similarity}")
    
    # Calculate cosine similarity
    return similarity 