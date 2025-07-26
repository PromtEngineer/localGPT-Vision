import os
from typing import Optional, List
import mlx.core as mx
from mlx_lm import load, generate

# Available LLM models for text correction
AVAILABLE_LLM_MODELS = {
    # Small & Fast Models (< 2GB RAM)
    "Qwen2.5-0.5B": {
        "repo": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "description": "Tiny & fast (0.5B)",
        "size": "0.5B"
    },
    "gemma-2b": {
        "repo": "mlx-community/gemma-2-2b-it-4bit",
        "description": "Google's efficient (2B)",
        "size": "2B"
    },
    
    # Medium Models (2-4GB RAM)
    "Phi-3.5-mini": {
        "repo": "mlx-community/Phi-3.5-mini-instruct-4bit",
        "description": "Best balance (3.8B)",
        "size": "3.8B"
    },
    "gemma-3-4b": {
        "repo": "mlx-community/gemma-3-4b-it-qat-4bit",
        "description": "Google's latest (4B)",
        "size": "4B"
    },
    
    # Larger Models (4-6GB RAM)
    "Mistral-7B": {
        "repo": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "description": "Very capable (7B)",
        "size": "7B"
    },
    "Llama-3.1-8B": {
        "repo": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        "description": "State of the art (8B)",
        "size": "8B"
    },
}

class TextCorrector:
    """LLM-based text correction for transcriptions"""
    
    def __init__(self, model_name: str = "Phi-3.5-mini"):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.selected_model = model_name
        self._load_model(model_name)
    
    def _load_model(self, model_name: str):
        """Load the specified LLM model"""
        if model_name not in AVAILABLE_LLM_MODELS:
            print(f"Unknown model {model_name}, using default Phi-3.5-mini")
            model_name = "Phi-3.5-mini"
        
        model_info = AVAILABLE_LLM_MODELS[model_name]
        repo_id = model_info["repo"]
        
        try:
            print(f"Loading text correction model: {model_name} ({repo_id})")
            self.model, self.tokenizer = load(repo_id)
            self.model_name = model_name
            print(f"Successfully loaded {model_name}")
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            # Try fallback to Phi-3.5-mini if different model was selected
            if model_name != "Phi-3.5-mini":
                print("Falling back to Phi-3.5-mini")
                self._load_model("Phi-3.5-mini")
            else:
                print("Warning: No text correction model could be loaded")
    
    def set_model(self, model_name: str):
        """Change the LLM model"""
        if model_name != self.model_name:
            self._load_model(model_name)
    
    def correct(
        self, 
        text: str, 
        context: Optional[str] = None,
        remove_fillers: bool = True
    ) -> str:
        """Correct transcribed text"""
        if not self.model or not text or len(text.strip()) < 10:
            return text
        
        # Build correction prompt
        prompt = self._build_prompt(text, context, remove_fillers)
        
        try:
            # Generate correction
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=len(text) * 2
            )
            
            # Extract corrected text
            corrected = self._extract_correction(response)
            
            # Validate correction
            if self._is_valid_correction(text, corrected):
                return corrected
            else:
                return text
                
        except Exception as e:
            print(f"Text correction failed: {e}")
            return text
    
    def _build_prompt(
        self, 
        text: str, 
        context: Optional[str],
        remove_fillers: bool
    ) -> str:
        """Build the correction prompt"""
        base_prompt = """Fix ONLY spelling errors and remove filler words from this transcription.
DO NOT rephrase or summarize. Keep the original wording and structure.
Only fix obvious errors like duplicated words, misspellings, and remove filler words (um, uh, etc).
Output ONLY the cleaned text, nothing else."""
        
        if context:
            base_prompt = f"Context: {context}\n\n{base_prompt}"
        
        return f"""{base_prompt}

Original: {text}

Cleaned:"""
    
    def _extract_correction(self, response: str) -> str:
        """Extract the corrected text from model response"""
        # Remove any explanation or metadata
        lines = response.strip().split('\n')
        
        # Find the actual correction
        corrected_text = ""
        for line in lines:
            # Skip meta lines
            if any(marker in line.lower() for marker in ['original:', 'cleaned:', 'corrected:']):
                continue
            if line.strip():
                corrected_text = line.strip()
                break
        
        return corrected_text
    
    def _is_valid_correction(self, original: str, corrected: str) -> bool:
        """Validate that correction is reasonable"""
        if not corrected:
            return False
        
        # Check length difference (shouldn't be too different)
        len_ratio = len(corrected) / len(original)
        if len_ratio < 0.5 or len_ratio > 1.5:
            return False
        
        # Check word count difference
        orig_words = original.split()
        corr_words = corrected.split()
        word_ratio = len(corr_words) / len(orig_words)
        if word_ratio < 0.5 or word_ratio > 1.2:
            return False
        
        return True
    
    def get_filler_words(self) -> List[str]:
        """Get list of filler words to remove"""
        return [
            "um", "uh", "er", "ah", "like", "you know", "I mean",
            "actually", "basically", "literally", "right", "so"
        ]