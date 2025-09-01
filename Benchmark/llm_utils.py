import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
import dashscope
from google import genai
from openai import OpenAI
from typing import Optional, Dict, Any

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
    @staticmethod
    def format(text, color):
        return f"{color}{text}{Colors.END}"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QwenLLM:
    """Minimal Qwen LLM client for generating benchmark questions"""

    def __init__(self, api_key=None, model="qwen-max"):
        # Try to load API key from environment files
        parent_env_path = Path(__file__).parent.parent / ".env"
        parent_env_local_path = Path(__file__).parent.parent / ".env.local"
        local_env_path = Path(__file__).parent / ".env"

        if local_env_path.exists():
            load_dotenv(dotenv_path=local_env_path)
        elif parent_env_local_path.exists():
            load_dotenv(dotenv_path=parent_env_local_path)
        elif parent_env_path.exists():
            load_dotenv(dotenv_path=parent_env_path)

        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        if not self.api_key:
            raise ValueError("DashScope API key is required. Please set DASHSCOPE_API_KEY or QWEN_API_KEY in your environment or .env file")

        self.model = model
        
        # Load LLM configuration
        self.random_seed = int(os.getenv("LLM_RANDOM_SEED", "42"))
        self.default_temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    def generate_response(
        self, 
        prompt: str, 
        system_message: str = "You are a helpful assistant.",
        temperature: Optional[float] = None,
        return_json: bool = False,
        debug: bool = False
    ) -> Optional[str]:
        """
        Generate response from Qwen LLM
        
        Args:
            prompt: User prompt
            system_message: System message for context
            temperature: Sampling temperature (0.0-1.0), uses default if None
            return_json: Whether to parse response as JSON
            debug: If True, print full prompt and response
            
        Returns:
            Response string or parsed JSON dict
        """
        try:
            # Use default temperature if not specified
            if temperature is None:
                temperature = self.default_temperature
            
            if debug:
                print(f"\n{Colors.format('='*80, Colors.HEADER)}")
                print(f"{Colors.format('[DEBUG]', Colors.BOLD)} Model: {Colors.format(self.model, Colors.CYAN)}")
                print(f"{Colors.format('[DEBUG]', Colors.BOLD)} Temperature: {Colors.format(str(temperature), Colors.YELLOW)}, Seed: {Colors.format(str(self.random_seed), Colors.YELLOW)}")
                print(f"{Colors.format('='*80, Colors.HEADER)}")
                print(f"{Colors.format('[SYSTEM PROMPT]:', Colors.GREEN + Colors.BOLD)}")
                print(f"{Colors.format(system_message, Colors.GREEN)}")
                print(f"{Colors.format('-'*80, Colors.BLUE)}")
                print(f"{Colors.format('[USER PROMPT]:', Colors.CYAN + Colors.BOLD)}")
                print(f"{Colors.format(prompt, Colors.CYAN)}")
                print(f"{Colors.format('='*80, Colors.HEADER)}")
                input(f"{Colors.format('[DEBUG]', Colors.BOLD)} Press {Colors.format('Enter', Colors.YELLOW)} to send request...")
                
            logger.info(f"Sending prompt to {self.model} (temp={temperature}, seed={self.random_seed})")
            
            response = dashscope.Generation.call(
                api_key=self.api_key,
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                result_format="message",
                temperature=temperature,
                seed=self.random_seed,
            )

            if response.status_code == 200:
                content = response.output.choices[0].message.content
                logger.info("Successfully received response")
                
                if debug:
                    print(f"\n{Colors.format('[RESPONSE]:', Colors.RED + Colors.BOLD)}")
                    print(f"{Colors.format(content, Colors.RED)}")
                    print(f"{Colors.format('='*80, Colors.HEADER)}")
                    input(f"{Colors.format('[DEBUG]', Colors.BOLD)} Press {Colors.format('Enter', Colors.YELLOW)} to continue...")
                
                if return_json:
                    try:
                        cleaned_json = self._clean_json_string(content)
                        logger.info(f"Attempting to parse JSON: {cleaned_json[:200]}...")
                        return json.loads(cleaned_json)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON response: {e}")
                        logger.error(f"Raw response: {content}")
                        return None
                
                return content
            else:
                logger.error(f"API call failed with status: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error calling Qwen API: {e}")
            return None

    def _clean_json_string(self, json_str: str) -> str:
        """Extract JSON from response string"""
        # Try to find JSON array first
        array_start = json_str.find("[")
        array_end = json_str.rfind("]") + 1
        if array_start >= 0 and array_end > array_start:
            return json_str[array_start:array_end]
        
        # Fall back to JSON object
        obj_start = json_str.find("{")
        obj_end = json_str.rfind("}") + 1
        if obj_start >= 0 and obj_end > obj_start:
            return json_str[obj_start:obj_end]
        
        return json_str


class GeminiLLM:
    """Minimal Gemini LLM client for generating benchmark questions"""

    def __init__(self, api_key=None, model="gemini-1.5-flash"):
        # Try to load API key from environment files
        parent_env_path = Path(__file__).parent.parent / ".env"
        parent_env_local_path = Path(__file__).parent.parent / ".env.local"
        local_env_path = Path(__file__).parent / ".env"

        if local_env_path.exists():
            load_dotenv(dotenv_path=local_env_path)
        elif parent_env_local_path.exists():
            load_dotenv(dotenv_path=parent_env_local_path)
        elif parent_env_path.exists():
            load_dotenv(dotenv_path=parent_env_path)

        # Set API key in environment for genai client
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key is required. Please set GEMINI_API_KEY in your environment or .env file")
        
        os.environ["GEMINI_API_KEY"] = api_key
        self.client = genai.Client()
        self.model = model
        
        # Load LLM configuration
        self.random_seed = int(os.getenv("LLM_RANDOM_SEED", "42"))
        self.default_temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    def generate_response(
        self, 
        prompt: str, 
        system_message: str = "You are a helpful assistant.",
        temperature: Optional[float] = None,
        return_json: bool = False,
        debug: bool = False
    ) -> Optional[str]:
        """
        Generate response from Gemini LLM
        
        Args:
            prompt: User prompt
            system_message: System message for context
            temperature: Sampling temperature (0.0-1.0), uses default if None
            return_json: Whether to parse response as JSON
            debug: If True, print full prompt and response
            
        Returns:
            Response string or parsed JSON dict
        """
        try:
            # Use default temperature if not specified
            if temperature is None:
                temperature = self.default_temperature
            
            if debug:
                print(f"\n{Colors.format('='*80, Colors.HEADER)}")
                print(f"{Colors.format('[DEBUG]', Colors.BOLD)} Model: {Colors.format(self.model, Colors.CYAN)}")
                print(f"{Colors.format('[DEBUG]', Colors.BOLD)} Temperature: {Colors.format(str(temperature), Colors.YELLOW)}")
                print(f"{Colors.format('='*80, Colors.HEADER)}")
                print(f"{Colors.format('[SYSTEM PROMPT]:', Colors.GREEN + Colors.BOLD)}")
                print(f"{Colors.format(system_message, Colors.GREEN)}")
                print(f"{Colors.format('-'*80, Colors.BLUE)}")
                print(f"{Colors.format('[USER PROMPT]:', Colors.CYAN + Colors.BOLD)}")
                print(f"{Colors.format(prompt, Colors.CYAN)}")
                print(f"{Colors.format('='*80, Colors.HEADER)}")
                input(f"{Colors.format('[DEBUG]', Colors.BOLD)} Press {Colors.format('Enter', Colors.YELLOW)} to send request...")
                
            logger.info(f"Sending prompt to {self.model} (temp={temperature})")
            
            # Combine system message and user prompt
            full_prompt = f"{system_message}\n\n{prompt}"
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config={
                    "temperature": temperature,
                    "seed": self.random_seed
                }
            )

            if response and response.text:
                content = response.text
                logger.info("Successfully received response")
                
                if debug:
                    print(f"\n{Colors.format('[RESPONSE]:', Colors.RED + Colors.BOLD)}")
                    print(f"{Colors.format(content, Colors.RED)}")
                    print(f"{Colors.format('='*80, Colors.HEADER)}")
                    input(f"{Colors.format('[DEBUG]', Colors.BOLD)} Press {Colors.format('Enter', Colors.YELLOW)} to continue...")
                
                if return_json:
                    try:
                        cleaned_json = self._clean_json_string(content)
                        logger.info(f"Attempting to parse JSON: {cleaned_json[:200]}...")
                        return json.loads(cleaned_json)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON response: {e}")
                        logger.error(f"Raw response: {content}")
                        return None
                
                return content
            else:
                logger.error("API call failed - no response received")
                return None

        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return None

    def _clean_json_string(self, json_str: str) -> str:
        """Extract JSON from response string"""
        # Try to find JSON array first
        array_start = json_str.find("[")
        array_end = json_str.rfind("]") + 1
        if array_start >= 0 and array_end > array_start:
            return json_str[array_start:array_end]
        
        # Fall back to JSON object
        obj_start = json_str.find("{")
        obj_end = json_str.rfind("}") + 1
        if obj_start >= 0 and obj_end > obj_start:
            return json_str[obj_start:obj_end]
        
        return json_str


class LlamaLLM:
    """Minimal Llama LLM client for generating benchmark questions"""

    def __init__(self, api_key=None, model="meta-llama/llama-3.3-70b-instruct"):
        # Try to load API key from environment files
        parent_env_path = Path(__file__).parent.parent / ".env"
        parent_env_local_path = Path(__file__).parent.parent / ".env.local"
        local_env_path = Path(__file__).parent / ".env"

        if local_env_path.exists():
            load_dotenv(dotenv_path=local_env_path)
        elif parent_env_local_path.exists():
            load_dotenv(dotenv_path=parent_env_local_path)
        elif parent_env_path.exists():
            load_dotenv(dotenv_path=parent_env_path)

        api_key = api_key or os.getenv("LLAMA_API_KEY")
        if not api_key:
            raise ValueError("Llama API key is required. Please set LLAMA_API_KEY in your environment or .env file")
        
        self.client = OpenAI(
            base_url="https://api.novita.ai/openai",
            api_key=api_key,
        )
        self.model = model
        
        # Load LLM configuration
        self.random_seed = int(os.getenv("LLM_RANDOM_SEED", "42"))
        self.default_temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    def generate_response(
        self, 
        prompt: str, 
        system_message: str = "You are a helpful assistant.",
        temperature: Optional[float] = None,
        return_json: bool = False,
        debug: bool = False
    ) -> Optional[str]:
        """
        Generate response from Llama LLM
        
        Args:
            prompt: User prompt
            system_message: System message for context
            temperature: Sampling temperature (0.0-1.0), uses default if None
            return_json: Whether to parse response as JSON
            debug: If True, print full prompt and response
            
        Returns:
            Response string or parsed JSON dict
        """
        try:
            # Use default temperature if not specified
            if temperature is None:
                temperature = self.default_temperature
            
            if debug:
                print(f"\n{Colors.format('='*80, Colors.HEADER)}")
                print(f"{Colors.format('[DEBUG]', Colors.BOLD)} Model: {Colors.format(self.model, Colors.CYAN)}")
                print(f"{Colors.format('[DEBUG]', Colors.BOLD)} Temperature: {Colors.format(str(temperature), Colors.YELLOW)}, Seed: {Colors.format(str(self.random_seed), Colors.YELLOW)}")
                print(f"{Colors.format('='*80, Colors.HEADER)}")
                print(f"{Colors.format('[SYSTEM PROMPT]:', Colors.GREEN + Colors.BOLD)}")
                print(f"{Colors.format(system_message, Colors.GREEN)}")
                print(f"{Colors.format('-'*80, Colors.BLUE)}")
                print(f"{Colors.format('[USER PROMPT]:', Colors.CYAN + Colors.BOLD)}")
                print(f"{Colors.format(prompt, Colors.CYAN)}")
                print(f"{Colors.format('='*80, Colors.HEADER)}")
                input(f"{Colors.format('[DEBUG]', Colors.BOLD)} Press {Colors.format('Enter', Colors.YELLOW)} to send request...")
                
            logger.info(f"Sending prompt to {self.model} (temp={temperature}, seed={self.random_seed})")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                seed=self.random_seed,
                stream=False
            )

            if response and response.choices:
                content = response.choices[0].message.content
                logger.info("Successfully received response")
                
                if debug:
                    print(f"\n{Colors.format('[RESPONSE]:', Colors.RED + Colors.BOLD)}")
                    print(f"{Colors.format(content, Colors.RED)}")
                    print(f"{Colors.format('='*80, Colors.HEADER)}")
                    input(f"{Colors.format('[DEBUG]', Colors.BOLD)} Press {Colors.format('Enter', Colors.YELLOW)} to continue...")
                
                if return_json:
                    try:
                        cleaned_json = self._clean_json_string(content)
                        logger.info(f"Attempting to parse JSON: {cleaned_json[:200]}...")
                        return json.loads(cleaned_json)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON response: {e}")
                        logger.error(f"Raw response: {content}")
                        return None
                
                return content
            else:
                logger.error("API call failed - no response received")
                return None

        except Exception as e:
            logger.error(f"Error calling Llama API: {e}")
            return None

    def _clean_json_string(self, json_str: str) -> str:
        """Extract JSON from response string"""
        # Try to find JSON array first
        array_start = json_str.find("[")
        array_end = json_str.rfind("]") + 1
        if array_start >= 0 and array_end > array_start:
            return json_str[array_start:array_end]
        
        # Fall back to JSON object
        obj_start = json_str.find("{")
        obj_end = json_str.rfind("}") + 1
        if obj_start >= 0 and obj_end > obj_start:
            return json_str[obj_start:obj_end]
        
        return json_str


class DeepSeekLLM:
    """Minimal DeepSeek LLM client for generating benchmark questions"""

    def __init__(self, api_key=None, model="deepseek-v3.1"):
        # Try to load API key from environment files
        parent_env_path = Path(__file__).parent.parent / ".env"
        parent_env_local_path = Path(__file__).parent.parent / ".env.local"
        local_env_path = Path(__file__).parent / ".env"

        if local_env_path.exists():
            load_dotenv(dotenv_path=local_env_path)
        elif parent_env_local_path.exists():
            load_dotenv(dotenv_path=parent_env_local_path)
        elif parent_env_path.exists():
            load_dotenv(dotenv_path=parent_env_path)

        # Use QWEN_API_KEY as specified
        api_key = api_key or os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DeepSeek API key is required. Please set QWEN_API_KEY or DASHSCOPE_API_KEY in your environment or .env file")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model
        
        # Load LLM configuration
        self.random_seed = int(os.getenv("LLM_RANDOM_SEED", "42"))
        self.default_temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    def generate_response(
        self, 
        prompt: str, 
        system_message: str = "You are a helpful assistant.",
        temperature: Optional[float] = None,
        return_json: bool = False,
        debug: bool = False
    ) -> Optional[str]:
        """
        Generate response from DeepSeek LLM
        
        Args:
            prompt: User prompt
            system_message: System message for context
            temperature: Sampling temperature (0.0-1.0), uses default if None
            return_json: Whether to parse response as JSON
            debug: If True, print full prompt and response
            
        Returns:
            Response string or parsed JSON dict
        """
        try:
            # Use default temperature if not specified
            if temperature is None:
                temperature = self.default_temperature
            
            if debug:
                print(f"\n{Colors.format('='*80, Colors.HEADER)}")
                print(f"{Colors.format('[DEBUG]', Colors.BOLD)} Model: {Colors.format(self.model, Colors.CYAN)}")
                print(f"{Colors.format('[DEBUG]', Colors.BOLD)} Temperature: {Colors.format(str(temperature), Colors.YELLOW)}, Seed: {Colors.format(str(self.random_seed), Colors.YELLOW)}")
                print(f"{Colors.format('='*80, Colors.HEADER)}")
                print(f"{Colors.format('[SYSTEM PROMPT]:', Colors.GREEN + Colors.BOLD)}")
                print(f"{Colors.format(system_message, Colors.GREEN)}")
                print(f"{Colors.format('-'*80, Colors.BLUE)}")
                print(f"{Colors.format('[USER PROMPT]:', Colors.CYAN + Colors.BOLD)}")
                print(f"{Colors.format(prompt, Colors.CYAN)}")
                print(f"{Colors.format('='*80, Colors.HEADER)}")
                input(f"{Colors.format('[DEBUG]', Colors.BOLD)} Press {Colors.format('Enter', Colors.YELLOW)} to send request...")
                
            logger.info(f"Sending prompt to {self.model} (temp={temperature}, seed={self.random_seed})")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                seed=self.random_seed
            )

            if response and response.choices:
                content = response.choices[0].message.content
                logger.info("Successfully received response")
                
                if debug:
                    print(f"\n{Colors.format('[RESPONSE]:', Colors.RED + Colors.BOLD)}")
                    print(f"{Colors.format(content, Colors.RED)}")
                    print(f"{Colors.format('='*80, Colors.HEADER)}")
                    input(f"{Colors.format('[DEBUG]', Colors.BOLD)} Press {Colors.format('Enter', Colors.YELLOW)} to continue...")
                
                if return_json:
                    try:
                        cleaned_json = self._clean_json_string(content)
                        logger.info(f"Attempting to parse JSON: {cleaned_json[:200]}...")
                        return json.loads(cleaned_json)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON response: {e}")
                        logger.error(f"Raw response: {content}")
                        return None
                
                return content
            else:
                logger.error("API call failed - no response received")
                return None

        except Exception as e:
            logger.error(f"Error calling DeepSeek API: {e}")
            return None

    def _clean_json_string(self, json_str: str) -> str:
        """Extract JSON from response string"""
        # Try to find JSON array first
        array_start = json_str.find("[")
        array_end = json_str.rfind("]") + 1
        if array_start >= 0 and array_end > array_start:
            return json_str[array_start:array_end]
        
        # Fall back to JSON object
        obj_start = json_str.find("{")
        obj_end = json_str.rfind("}") + 1
        if obj_start >= 0 and obj_end > obj_start:
            return json_str[obj_start:obj_end]
        
        return json_str


def main():
    """Test the LLM classes"""
    # Test QwenLLM
    print("Testing QwenLLM...")
    try:
        llm = QwenLLM()
        
        # Test simple text generation
        prompt = "Generate a brief question about urban zoning policies."
        response = llm.generate_response(prompt)
        print(f"Qwen Response: {response}")
        
        # Test JSON generation
        json_prompt = """
        Generate a simple question-answer pair about housing policy in JSON format:
        {
            "question": "your question here",
            "answer": "your answer here"
        }
        """
        json_response = llm.generate_response(
            json_prompt, 
            return_json=True
        )
        print(f"Qwen JSON Response: {json_response}")
        
    except ValueError as e:
        print(f"Qwen Error: {e}")
        print("Please set DASHSCOPE_API_KEY or QWEN_API_KEY in your environment or .env file")
    
    # Test GeminiLLM
    print("\nTesting GeminiLLM...")
    try:
        gemini_llm = GeminiLLM()
        
        # Test simple text generation
        prompt = "Generate a brief question about urban zoning policies."
        response = gemini_llm.generate_response(prompt)
        print(f"Gemini Response: {response}")
        
        # Test JSON generation
        json_prompt = """
        Generate a simple question-answer pair about housing policy in JSON format:
        {
            "question": "your question here",
            "answer": "your answer here"
        }
        """
        json_response = gemini_llm.generate_response(
            json_prompt, 
            return_json=True
        )
        print(f"Gemini JSON Response: {json_response}")
        
    except ValueError as e:
        print(f"Gemini Error: {e}")
        print("Please set GEMINI_API_KEY in your environment or .env file")
    
    # Test LlamaLLM
    print("\nTesting LlamaLLM...")
    try:
        llama_llm = LlamaLLM()
        
        # Test simple text generation
        prompt = "Generate a brief question about urban zoning policies."
        response = llama_llm.generate_response(prompt)
        print(f"Llama Response: {response}")
        
        # Test JSON generation
        json_prompt = """
        Generate a simple question-answer pair about housing policy in JSON format:
        {
            "question": "your question here",
            "answer": "your answer here"
        }
        """
        json_response = llama_llm.generate_response(
            json_prompt, 
            return_json=True
        )
        print(f"Llama JSON Response: {json_response}")
        
    except ValueError as e:
        print(f"Llama Error: {e}")
        print("Please set LLAMA_API_KEY in your environment or .env file")
    
    # Test DeepSeekLLM
    print("\nTesting DeepSeekLLM...")
    try:
        deepseek_llm = DeepSeekLLM()
        
        # Test simple text generation
        prompt = "Generate a brief question about urban zoning policies."
        response = deepseek_llm.generate_response(prompt)
        print(f"DeepSeek Response: {response}")
        
        # Test JSON generation
        json_prompt = """
        Generate a simple question-answer pair about housing policy in JSON format:
        {
            "question": "your question here",
            "answer": "your answer here"
        }
        """
        json_response = deepseek_llm.generate_response(
            json_prompt, 
            return_json=True
        )
        print(f"DeepSeek JSON Response: {json_response}")
        
    except ValueError as e:
        print(f"DeepSeek Error: {e}")
        print("Please set QWEN_API_KEY or DASHSCOPE_API_KEY in your environment or .env file")


if __name__ == "__main__":
    main()
