# models/model_loader.py

import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import MllamaForConditionalGeneration
from vllm.sampling_params import SamplingParams
from transformers import AutoModelForCausalLM
import google.generativeai as genai
from vllm import LLM

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from logger import get_logger

logger = get_logger(__name__)

# Cache for loaded models
_model_cache = {}

def detect_device():
    """
    Detects the best available device (CUDA, MPS, or CPU).
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def load_model(model_choice):
    """
    Loads and caches the specified model.
    """
    global _model_cache

    if model_choice in _model_cache:
        logger.info(f"Model '{model_choice}' loaded from cache.")
        return _model_cache[model_choice]

    if model_choice == 'qwen':
        device = detect_device()
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.float16 if device != 'cpu' else torch.float32,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        model.to(device)
        _model_cache[model_choice] = (model, processor, device)
        logger.info("Qwen model loaded and cached.")
        return _model_cache[model_choice]

    elif model_choice == 'gemini':
        # Load Gemini model
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-002')  # Use the appropriate model name
        return model, None


    elif model_choice == 'llama-vision':
        # Load Llama-Vision model
        device = detect_device()
        # model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        model_id = "alpindale/Llama-3.2-11B-Vision-Instruct"
        model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device != 'cpu' else torch.float32,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_id)
        model.to(device)
        _model_cache[model_choice] = (model, processor, device)
        logger.info("Llama-Vision model loaded and cached.")
        return _model_cache[model_choice]
    
    elif model_choice == "pixtral":
        device = detect_device()
        model = LLM(model="mistralai/Pixtral-12B-2409", 
                    tokenizer_mode="mistral",                 
                    gpu_memory_utilization=0.8,  # Increase GPU memory utilization
                    max_model_len=8192,  # Decrease max model length
                    dtype="float16",  # Use half precision to save memory
                    trust_remote_code=True)
        sampling_params = SamplingParams(max_tokens=1024)
        _model_cache[model_choice] = (model, sampling_params, device)
        return _model_cache[model_choice]
    
    elif model_choice == "molmo":
        device = detect_device()
        processor = AutoProcessor.from_pretrained(
            'allenai/MolmoE-1B-0924',
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )
        model = AutoModelForCausalLM.from_pretrained(
            'allenai/MolmoE-1B-0924',
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )
        _model_cache[model_choice] = (model, processor, device)
        return _model_cache[model_choice]
    else:
        logger.error(f"Invalid model choice: {model_choice}")
        raise ValueError("Invalid model choice.")
