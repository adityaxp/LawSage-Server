from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
import torch
from logger import logtool
import re
from RAG_central_acts import RAG_central_acts_get_context
from RAG_constitution import RAG_constitution_get_context
from huggingface_hub import login

login("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")

load_model = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logtool.write_log(f"Using device: {str(device)}", "RAG")

system_prompt = "You are an AI powered legal advisory chatbot named Law Sage. Only provide responses in a correct leagal context. Please refrain from referencing this unless specifically it is asked."
model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
hf_auth = "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"

if load_model:
    logtool.write_log("Loading model in nf4", "RAG")
    
    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
        llm_int8_enable_fp32_cpu_offload=True
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token = hf_auth,
        cache_dir="models/mistralai/Mixtral-8x7B-Instruct-v0.1"
    )

    logtool.write_log(f"Using model : {model_name}", "RAG")

    logtool.write_log("Loading tokenizer", "RAG")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.config.update({"max_position_embeddings": 512, "max_length": 512})
    model.resize_token_embeddings(len(tokenizer))

    logtool.write_log("Creating RAG pipeline", "RAG")

    RAG_pipeline = pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=True,
        task='text-generation',
        max_new_tokens=512,
        repetition_penalty=1.1
    )
else:
    logtool.write_log("Model already loaded", "RAG")
    logtool.write_log(f"Using model : {model_name}", "RAG")



def parse_generated_text(generated_text):
    generated_text_value = generated_text['generated_text']
    inst_pattern = r'\[INST\](.*?)\[/INST\]'
    inst_matches = re.findall(inst_pattern, generated_text_value)
    remaining_text = re.sub(inst_pattern, '', generated_text_value)
    return inst_matches, remaining_text.strip()


def get_RAG_response(query, RAG_type):
    logtool.write_log("Generating RAG response", "RAG")
    results = []
    if RAG_type == "RAG_constitution":
        results = RAG_constitution_get_context(query)
    elif RAG_type == "RAG_central_acts":
        results = RAG_central_acts_get_context(query)
    else:
        return "Error: Undefined RAG Type"
    ref = results[0]
    answer = RAG_pipeline(f"""[INST]  <<SYS>> {system_prompt} <</SYS>> Answer this question {query} based on the given context {ref} [/INST]""")
    generated_text_input = answer[0]
    logtool.write_log("Parsing response", "RAG")
    inst_contents, RAG_response = parse_generated_text(generated_text_input)
    return RAG_response



query = "Give preamble of india constitution"
print(get_RAG_response(query, "RAG_constitution"))