try:
    import os
    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def download_model(model_name, hf_token, models_dir="./pleias_models"):
        """
        Download a PleIAs model for RAG if not already present.
        
        Args:
            model_name: Name of the model to download. Available options:
                        - "1b_rag": PleIAs/1b_rag_traceback
            hf_token: Hugging Face API token
            models_dir: Directory where models will be stored (default: ./pleias_models)
            
        Returns:
            Path to the downloaded model
        
        Raises:
            ValueError: If model_name is not recognized
        """
        # Dictionary of available models
        AVAILABLE_MODELS = {
            "1b_rag": "PleIAs/1b_rag_traceback",
            # Add more models as they become available
        }
        
        # Check if model name is valid
        if model_name not in AVAILABLE_MODELS:
            available = ", ".join(AVAILABLE_MODELS.keys())
            raise ValueError(f"Model '{model_name}' not found. Available models: {available}")
        
        # Get the HF model ID
        hf_model_id = AVAILABLE_MODELS[model_name]
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Model-specific directory
        model_dir = os.path.join(models_dir, model_name)
        
        # Check if model is already downloaded by looking for config.json
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            print(f"Model '{model_name}' already downloaded at {model_dir}")
            return model_dir
        
        # If we get here, we need to download the model
        os.makedirs(model_dir, exist_ok=True)
        print(f"Downloading model '{model_name}' from '{hf_model_id}'...")
        
        # First, just save the tokenizer which requires minimal RAM
        print("Saving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id, token=hf_token)
        tokenizer.save_pretrained(model_dir)
        del tokenizer
        gc.collect()  # Force garbage collection
        print("Tokenizer saved successfully")
        
        # Load model with memory optimization settings
        print("Loading model (this might take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            token=hf_token,
            device_map="auto",  # Automatically distribute across available devices
            offload_folder="offload",  # Use disk offloading if needed
            torch_dtype=torch.float16,  # Use half precision to save memory
            low_cpu_mem_usage=True  # Optimize for low CPU memory usage
        )
        
        # Save model in a memory-efficient way
        print("Saving model...")
        model.save_pretrained(
            model_dir,
            safe_serialization=True,  # Use safetensors format
            max_shard_size="2GB"  # Split into 2GB chunks for easier handling
        )
        
        # Clean up
        del model
        gc.collect()
        torch.cuda.empty_cache()  # If using GPU
        
        print(f"Model '{model_name}' successfully saved to {model_dir}")
        
        return model_dir
except ImportError:
    ImportError("The necessary libraries (torch and/or transformers) are not installed. Please check the optional dependencies.")