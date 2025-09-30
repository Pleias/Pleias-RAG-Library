from typing import Dict, List, Any, Optional, Literal
import re
try:
    import torch
except ImportError:
    print("torch is not installed. Check optional dependencies if you need it.")
import json

class RAGWithCitations:
    def __init__(
        self,
        model_path_or_name: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        trust_remote_code: bool = True,
        hf_token=None,
        models_dir="./pleias_models",
        backend: Optional[Literal["vllm", "transformers", "llama_cpp", "lmstudio"]] = None
    ):
        """
        Initialize the RAG Generator with either vLLM (if CUDA available) or transformers.
        
        Args:
            model_path_or_name: Path to the model, HuggingFace model name, or name from available models:
                               - "1b_rag": PleIAs/1b_rag_traceback
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_p: Top-p sampling parameter (lower = more focused)
            repetition_penalty: Repetition penalty to avoid loops
            trust_remote_code: Whether to trust remote code in model repo
            hf_token: Hugging Face API token (required if using predefined model names)
            models_dir: Directory where models will be stored (default: ./pleias_models)
            backend: Optional backend to use for model loading. If not specified, will use vLLM if CUDA is available and transformers otherwise.
        """
        # Check if this is a predefined model name
        AVAILABLE_MODELS = {
            "1b_rag": "PleIAs/1b_rag_traceback",
            # Add more models as they become available
        }
        
        if model_path_or_name in AVAILABLE_MODELS:
            # Try to use the download_model function if available
            try:
                from .model_downloader import download_model
                if hf_token is None:
                    raise ValueError("HF token is required to download models from Hugging Face")
                model_path = download_model(model_path_or_name, hf_token, models_dir)
                model_path_or_name = model_path
                print(f"Using model from: {model_path}")
            except ImportError:
                # If the module isn't available, use the direct model name
                print(f"Model downloader not available, using HF model name directly: {AVAILABLE_MODELS[model_path_or_name]}")
                model_path_or_name = AVAILABLE_MODELS[model_path_or_name]
        
        self.model_path = model_path_or_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.trust_remote_code = trust_remote_code

        try:
            # Check if CUDA is available
            self.cuda_available = torch.cuda.is_available()
            print(f"CUDA available: {self.cuda_available}")
        except ImportError:
            print("torch is not installed. Check optional dependencies if you need it.")

        backends_init = {
            "vllm": self._init_vllm, 
            "transformers": self._init_transformers,
            "llama_cpp": self._init_llama_cpp,
            "lmstudio": self._init_lmstudio         
            }
        chosen_backend = backend or ("vllm" if self.cuda_available else "transformers")
        self.backend = chosen_backend
        backends_init[chosen_backend]()


    #################################
    # Model Initialization Methods  #
    #################################

    def _init_vllm(self):
        """
        Initialize using vLLM for GPU acceleration.
        This method sets up the LLM with vLLM backend for faster inference on GPU.
        """
        from vllm import LLM, SamplingParams
        import torch

        print(f"Loading model with vLLM from {self.model_path}...")
        self.llm = LLM(
            model=self.model_path,
            trust_remote_code=self.trust_remote_code,
            dtype=torch.float16  # Use float16 instead of bfloat16 for Tesla T4 compatibility, consider replacing with bfloat16 once we host elsewhere
        )
        print("Model loaded successfully with vLLM")

        # Get tokenizer from the loaded model
        tokenizer = self.llm.get_tokenizer()

        # Set up sampling parameters for vLLM
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            repetition_penalty=self.repetition_penalty,
            skip_special_tokens=False,
            stop_token_ids=[tokenizer.eos_token_id]
        )

    def _init_transformers(self):
        """
        Initialize using transformers for CPU fallback.
        This method sets up the model and tokenizer using HuggingFace Transformers
        when GPU acceleration is not available.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model with transformers from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",  # Will use CPU if no GPU is available
            trust_remote_code=self.trust_remote_code
        )
        print("Model loaded successfully with transformers")
        
    def _init_llama_cpp(self):
        """
        Initialize the model using llama_cpp
        """
        from llama_cpp import Llama
        
        print(f"Loading model with llama_cpp from {self.model_path}...")
        
        self.model = Llama(
            model_path=self.model_path,
            n_ctx=4096,            
            n_gpu_layers=0,    
            verbose=False,    
        )
        print("Model loaded successfully with llama_cpp")

    def _init_lmstudio(self):
        """
        Initialize the model using lmstudio.

        This pressuposes that lmstudio is listening on localhost:1234.
        Follow the naming convention of lmstudio. e.g., "pleias-rag-350m".
        """
        from lmstudio import llm
        
        print(f"Loading model with lmstudio from {self.model_path}...")

        self.model = llm(self.model_path)
        
        print("Model loaded successfully with lmstudio")

    ###################################
    # Prompt Formatting and Generation #
    ###################################

    def format_prompt(self, query: str, sources: List[Dict[str, Any]]) -> str:
        """
        Format the query and sources into a prompt with special tokens.

        The prompt follows a specific format with special tokens to guide the model:
        - <|query_start|>...<|query_end|> for the user's question
        - <|source_start|><|source_id|>N ...<|source_end|> for each source
        - <|language_start|> to indicate the beginning of generation
        Args:
            query: The user's question
            sources: List of source documents with their metadata. Format is list of dictionaries,
                     each with a "text" key and optional "metadata" key.
                     The metadata is not used in the prompt but can be useful for later processing.
                     Example: [{"text": "Document text", "metadata": {"source_id": 1, "source_name": "Doc1"}}]
        Returns:
            Formatted prompt string
        """
        prompt = f"<|query_start|>{query}<|query_end|>\n"

        # Add each source with its ID
        for idx, source in enumerate(sources, 1):
            source_text = source.get("text", "")
            prompt += f"<|source_start|><|source_id|>{idx} {source_text}<|source_end|>\n"

        # Add the source analysis start token
        prompt += "<|language_start|>\n"

        return prompt

    def _generate_vllm(self, formatted_prompt: str) -> str:
        """
        Generate text using vLLM backend.

        This method handles text generation when using the vLLM backend (GPU).

        Args:
            formatted_prompt: The properly formatted input prompt

        Returns:
            Generated text response
        """
        outputs = self.llm.generate(formatted_prompt, self.sampling_params)
        return outputs[0].outputs[0].text

    def _generate_transformers(self, formatted_prompt: str) -> str:
        """
        Generate text using transformers backend.

        This method handles text generation when using the Transformers backend (CPU).

        Args:
            formatted_prompt: The properly formatted input prompt

        Returns:
            Generated text response
        """
        input_ids = self.tokenizer(formatted_prompt, return_tensors="pt").input_ids

        # Move to GPU if available (though we're in this method because CUDA isn't available)
        device = self.model.device
        input_ids = input_ids.to(device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode only the newly generated tokens
        generated_text = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=False
        )

        return generated_text
    
    def _generate_llama_cpp(self, formatted_prompt: str) -> str:
        """
        Generate text using llama_cpp backend.
        This method handles text generation when using the llama_cpp backend (CPU)
        
        Args:
            formatted_prompt: The properly formatted input prompt

        Returns:
            Generated text response
        """
        
        tokens = self.model.generate(
                self.model.tokenize(formatted_prompt.encode("utf-8"), special=True), 
                temp=self.temperature,
                top_p=self.top_p,
                repeat_penalty=self.repetition_penalty,
                reset=True,
            )
        generated_text = ""

        for i, t in enumerate(tokens):
            piece = self.model.detokenize([t], special=True).decode("utf-8", errors="replace")    
            if (piece == "<|end_of_text|>") | (i >= self.max_tokens):
                break
            generated_text += piece
        
        return generated_text.strip()
    
    def _generate_lmstudio(self, formatted_prompt: str) -> str:
        """
        Generate text using lmstudio.
        
        Args:
            formatted_prompt: The properly formatted input prompt

        Returns:
            Generated text response
        """
        prediction = self.model.respond_stream(formatted_prompt)
        response = ""
        for token in prediction:
            response += token.content
        return response

    #############################
    # Response Processing       #
    #############################

    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract all sections from the generated text using the output format.

        The model's output is structured with section markers that need to be parsed.
        This method extracts different sections like query_analysis, query_report,
        source_analysis, draft, and answer.

        Note: query_analysis is included in the prompt, not in the output.
        Args:
            text: The generated text response
        Returns:
            Dictionary with all extracted sections
        """
        result = {}

        # For language, we need to handle it differently since it's in the prompt
        # Extract everything from the start until query_analysis_end
        language_end_match = re.search(r'<\|language_end\|>', text, re.DOTALL)
        if language_end_match:
            end_pos = language_end_match.start()
            result['language'] = text[:end_pos].strip()

        # Define other section patterns to extract
        section_patterns = {
            'query_analysis': r'<\|query_analysis_start\|>(.*?)<\|query_analysis_end\|>',
            'query_report': r'<\|query_report_start\|>(.*?)<\|query_report_end\|>',
            'source_analysis': r'<\|source_analysis_start\|>(.*?)<\|source_analysis_end\|>',
            'draft': r'<\|draft_start\|>(.*?)<\|draft_end\|>',
            'answer': r'<\|answer_start\|>(.*?)<\|answer_end\|>'
        }

        # Extract each section using regex
        for section_name, pattern in section_patterns.items():
            section_match = re.search(pattern, text, re.DOTALL)
            if section_match:
                result[section_name] = section_match.group(1).strip()

        # If no sections were found, return the full text
        if not result:
            result['full_text'] = text

        return result

    def extract_citations(self, answer: str, sources: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract citations from the answer and format them with numbered references.

        Args:
            answer: The answer text containing citations
            sources: List of source documents (optional)

        Returns:
            Dictionary with clean text and citations data
        """
        # Pattern to match <ref name="<|source_id|>NUMBER">text</ref>
        citation_pattern = r'<ref name="(?:<\|source_id\|>)?(\d+)">(.*?)<\/ref>'

        # Create a working copy and citation list
        clean_text = answer
        citations = []

        # Find all citations and process them one by one
        citation_count = 0
        while True:
            match = re.search(citation_pattern, clean_text)
            if not match:
                break

            citation_count += 1
            source_id = match.group(1)
            cited_text = match.group(2)
            full_match = match.group(0)
            start_pos = match.start()

            # Get some context for supported_text (look back up to 150 chars for context)
            text_before = clean_text[:start_pos]
            sentence_boundary = max(
                text_before.rfind('. '),
                text_before.rfind('! '),
                text_before.rfind('? '),
                text_before.rfind('\n')
            )

            if sentence_boundary == -1:
                supported_text = text_before[-min(150, len(text_before)):].strip()
            else:
                supported_text = text_before[sentence_boundary+2:].strip()

            # Replace this citation with a numbered reference
            clean_text = clean_text.replace(full_match, f"[{citation_count}]", 1)

            # Store citation data
            citations.append({
                "citation_number": citation_count,
                "source_id": source_id,
                "cited_text": cited_text,
                "supported_text": supported_text
            })

        # If no citations found, return the original text
        if not citations:
            return {
                "clean_text": answer,
                "citations": []
            }

        # Create citations section
        citations_section = "\n\n**Citations**\n"
        for citation in citations:
            citations_section += f"[{citation['citation_number']}] \"{citation['cited_text']}\" [Source {citation['source_id']}]\n"

        # Add citations section to clean text
        final_text = clean_text + citations_section

        return {
            "clean_text": final_text,
            "citations": citations
        }
    #############################
    # Output Handling Utilities #
    #############################

    def to_json(self, data: Dict[str, Any], indent: int = 2) -> str:
        """
        Convert data to a JSON string.
        Args:
            data: Dictionary to convert
            indent: Indentation level for pretty printing
        Returns:
            JSON string representation
        """
        return json.dumps(data, indent=indent)

    #############################
    # Main Interface Methods    #
    #############################

    def generate(self, query: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a response based on the query and sources.

        This is the main method to use for generating responses. It:
        1. Formats the prompt
        2. Generates text using the appropriate backend
        3. Processes the response to extract sections
        4. Extracts and formats citations
        Args:
            query: The user's question
            sources: List of source documents with their metadata
        Returns:
            Dictionary with raw response and processed sections
        """
        formatted_prompt = self.format_prompt(query, sources)

        # Generate response using the appropriate backend
        backends_generation = {
            "vllm": self._generate_vllm, 
            "transformers": self._generate_transformers,
            "llama_cpp": self._generate_llama_cpp,
            "lmstudio": self._generate_lmstudio
        }
        raw_response = backends_generation[self.backend](formatted_prompt)

        # Process the response
        sections = self.extract_sections(raw_response)

        # Extract citations if answer section exists
        if 'answer' in sections:
            citation_info = self.extract_citations(sections['answer'], sources)
            sections['clean_answer'] = citation_info['clean_text']
            sections['citations'] = citation_info['citations']

        response = {
            "raw_response": raw_response,
            "processed": sections,
            "backend_used": self.backend
        }

        return response

    def process_request(self, request_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request in JSON format.

        This method provides a JSON-based interface for the RAG system.
        It extracts the query and sources from the request JSON,
        processes them, and returns a structured response.
        Args:
            request_json: JSON with query and sources
        Returns:
            Response with generated text and processed sections
        """
        query = request_json.get("query", "")
        sources = request_json.get("sources", [])

        # Extract metadata from sources
        sources_with_metadata = []
        for idx, source in enumerate(sources, 1):
            metadata = source.get("metadata", {})
            # All sources are included, even if no metadata exists
            sources_with_metadata.append({
                "id": idx,
                "metadata": metadata  # This will be an empty dict if no metadata exists
            })

        # Generate response
        response_data = self.generate(query, sources)

        result = {
            "query": query,
            "sources_used": sources_with_metadata,
            "raw_response": response_data["raw_response"],
            "processed_response": response_data["processed"],
            "backend_used": response_data["backend_used"]
        }

        return result
