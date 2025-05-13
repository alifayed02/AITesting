import os
import re
import json
import time
from typing import Dict, Any, Optional, List
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys

def force_json(resp: str) -> Dict[str, Any]:
    """
    Extracts the first JSON object from the response text and parses it.
    If no JSON is found, attempts to convert the response to a JSON format
    based on common response patterns for multi-part answers (a, b, c).
    """
    match = re.search(r"\{.*\}", resp, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    # If no JSON found, try to parse the response into a structured format
    # Look for patterns like (a) answer (b) answer (c) answer
    result = {}
    
    # Try to extract answers for parts a, b, c, etc.
    parts = re.findall(r"\(?([a-z])\)?[.:]?\s*([^(]+?)(?=\([a-z]\)|$)", resp, re.IGNORECASE | re.DOTALL)
    
    if parts:
        for part, answer in parts:
            # Clean up the answer
            clean_answer = answer.strip()
            result[part.lower()] = clean_answer
        return result
    
    # If no parts were found, just return the raw text
    return {"raw": resp, "error": "Could not parse multi-part answer into structured format"}

class SeleniumLLMClient:
    def __init__(self):
        self.drivers = {}
        self.wait_time = 30  # seconds to wait for elements
        self.script_paths = {
            "gpt": "chatgpt.side",
            "deepseek": "deepseek.side",
            "perplexity": "perplexity.side"
        }
        
    def _get_driver(self, model: str) -> webdriver.Chrome:
        """Get or create a Chrome driver for the specified model."""
        if model not in self.drivers:
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')  # Run in headless mode
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            self.drivers[model] = webdriver.Chrome(options=options)
            
            # Load the Selenium IDE script
            with open(self.script_paths[model], 'r') as f:
                self.scripts[model] = json.load(f)
            
            # Execute the initial setup commands from the script
            for command in self.scripts[model]['tests'][0]['commands']:
                if command['command'] == 'open':
                    self.drivers[model].get(command['target'])
                elif command['command'] == 'setWindowSize':
                    width, height = map(int, command['target'].split('x'))
                    self.drivers[model].set_window_size(width, height)
            
            # Wait for the page to load
            time.sleep(5)  # Basic wait for initial load
            
        return self.drivers[model]
    
    def _execute_script_commands(self, driver: webdriver.Chrome, script: dict, question: str, image_url: Optional[str] = None) -> str:
        """Execute the commands from a Selenium IDE script."""
        response_text = ""
        
        for command in script['tests'][0]['commands']:
            try:
                if command['command'] == 'click':
                    # Find the element using the first available target
                    element = None
                    for target in command['targets']:
                        try:
                            if target[1] == 'css:finder':
                                element = WebDriverWait(driver, self.wait_time).until(
                                    EC.element_to_be_clickable((By.CSS_SELECTOR, target[0]))
                                )
                                break
                            elif target[1] == 'xpath:idRelative':
                                element = WebDriverWait(driver, self.wait_time).until(
                                    EC.element_to_be_clickable((By.XPATH, target[0]))
                                )
                                break
                        except:
                            continue
                    
                    if element:
                        element.click()
                
                elif command['command'] == 'editContent':
                    # This is where we input our question
                    element = WebDriverWait(driver, self.wait_time).until(
                        EC.presence_of_element_located((By.ID, command['target']))
                    )
                    element.clear()
                    
                    # If there's an image, upload it first
                    if image_url:
                        self._upload_image(driver, image_url)
                    
                    # Send the question
                    element.send_keys(question)
                
                elif command['command'] == 'mouseOver':
                    element = WebDriverWait(driver, self.wait_time).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, command['target']))
                    )
                    ActionChains(driver).move_to_element(element).perform()
                
                elif command['command'] == 'mouseOut':
                    element = WebDriverWait(driver, self.wait_time).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, command['target']))
                    )
                    ActionChains(driver).move_to_element(element).perform()
                
                elif command['command'] == 'runScript':
                    driver.execute_script(command['target'])
                
                # After clicking the submit button, wait for and capture the response
                if command['command'] == 'click' and 'submit' in command['target'].lower():
                    # Wait for the response to appear
                    response_element = WebDriverWait(driver, self.wait_time).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".markdown-content, .response-content"))
                    )
                    response_text = response_element.text
                
            except Exception as e:
                print(f"Warning: Failed to execute command {command['command']}: {e}")
                continue
        
        return response_text
    
    def _upload_image(self, driver: webdriver.Chrome, image_url: str) -> None:
        """Upload an image to the chat interface."""
        try:
            # Find the image upload button
            upload_button = WebDriverWait(driver, self.wait_time).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
            )
            
            # If it's a local file path, use it directly
            if os.path.exists(image_url):
                upload_button.send_keys(os.path.abspath(image_url))
            else:
                # If it's a URL, download it first
                import requests
                import tempfile
                
                response = requests.get(image_url)
                if response.status_code == 200:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                        temp_file.write(response.content)
                        upload_button.send_keys(temp_file.name)
                    os.unlink(temp_file.name)
                else:
                    raise Exception(f"Failed to download image from {image_url}")
            
            # Wait for the image to be uploaded
            time.sleep(2)
            
        except Exception as e:
            print(f"Warning: Failed to upload image: {e}")
    
    def get_response(
        self,
        question: str,
        model: str,
        numeric_only: bool = False,
        multi_part: bool = False,
        image_url: Optional[str] = None
    ) -> Any:
        """
        Get response from the specified model using Selenium.
        Supports both local file paths and URLs for images.
        """
        driver = self._get_driver(model)
        
        # Build the prompt with appropriate instructions
        if numeric_only:
            prompt = f"Answer with exactly the numeric result—no units, no explanation: {question}"
        elif multi_part:
            prompt = f"Answer in JSON format with keys 'a', 'b', 'c', etc. for each part: {question}"
        else:
            prompt = f"Answer in one clear sentence only—no bullet points, no extra elaboration: {question}"
        
        # Execute the script commands
        raw = self._execute_script_commands(driver, self.scripts[model], prompt, image_url)
        
        return force_json(raw) if multi_part else raw
    
    def get_all_responses(
        self,
        question: str,
        models: Optional[List[str]] = None,
        numeric_only: bool = False,
        multi_part: bool = False,
        image_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get responses from multiple models using Selenium.
        """
        if models is None:
            models = ["gpt", "deepseek", "perplexity"]
        
        responses: Dict[str, Any] = {}
        for m in models:
            try:
                responses[m] = self.get_response(question, m, numeric_only, multi_part, image_url)
            except Exception as e:
                responses[m] = f"ERROR: {e}"
        return responses
    
    def __del__(self):
        """Clean up all browser instances when the object is destroyed."""
        for driver in self.drivers.values():
            try:
                driver.quit()
            except:
                pass 