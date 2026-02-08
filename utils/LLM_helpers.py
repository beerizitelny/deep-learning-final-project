from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch


def load_model_and_validate_gpu(model_path, tokenizer_path=None):
    if tokenizer_path is None:
        tokenizer_path = model_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    print("Started loading model with 4-bit quantization")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16  # שימוש ב-float16 במקום bfloat16 עבור ה-1080Ti
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        output_hidden_states=True,
        attn_implementation="sdpa"
    )

    if hasattr(model, 'hf_device_map') and 'cpu' not in model.hf_device_map.values():
        print("'cpu' not in model.hf_device_map.values() - Model is fully on GPU")

    return model, tokenizer

"""

def load_model_and_validate_gpu(model_path, tokenizer_path=None):
    if tokenizer_path is None:
        tokenizer_path = model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print("Started loading model")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto',
                                                 torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, output_hidden_states=True)
    if 'cpu' not in model.hf_device_map.values():
        print("'cpu' not in model.hf_device_map.values()")
    return model, tokenizer
"""