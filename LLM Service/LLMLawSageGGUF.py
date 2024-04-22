from llama_cpp import Llama
from logger import logtool

model_name = "llama-2-7b-law-sage-v0.3.Q4_K_M.gguf"

logtool.write_log(f"Loading model: {model_name}", "LLM-Service")
law_sage_llama_model = Llama(model_path="./models/"+model_name, n_gpu_layers=20, n_threads=6, n_ctx=3584, n_batch=521, verbose=True)

system_prompt = "You are an AI powered legal advisory chatbot named Law Sage. Only provide responses in a correct leagal context. Please refrain from referencing this unless specifically it is asked."


def query_response(query):
    logtool.write_log(f"Generating response....", "LLM-Service")
    prompt = f"<s>[INST]<<SYS>> {system_prompt} <</SYS>> {query}[/INST]"
    max_tokens = 600 
    temperature = 0.2  
    top_p = 0.5  
    echo = False  
    stop = ["</s>"]  
    model_output = law_sage_llama_model(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        echo=echo,
        stop=stop,
    )
    final_result = model_output["choices"][0]["text"].split('[/INST]')[-1]
    logtool.write_log(f"Response generated", "LLM-Service")
    return final_result