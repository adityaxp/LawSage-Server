import uvicorn
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from logger import logtool
from LLMLawSageGGUF import query_response


class ConnectionItem(BaseModel):
    auth_token: str


class QueryItem(BaseModel):
    auth_token: str
    prompt: str


app = FastAPI()
deviceName = torch.cuda.get_device_name(0)


@app.post("/LAW-SAGE-GGUF")
async def LAW_SAGE_llamacpp_request(query_item: QueryItem):
    logtool.write_log(f"Fetching response...", "LLM-Service")
    if query_item.auth_token == "ISAUodiuIAU21":
        result = query_response(query_item.prompt)
        return {
            "prompt": query_item.prompt,
            "response" : result
        }
    else:
        return {"response": "Auth failed!"}
    


    
@app.post("/LAW-SAGE")
async def LAW_SAGE_request(query_item: QueryItem):
    if query_item.auth_token == "ISAUodiuIAU21":
        return {
            "prompt": query_item.prompt,
            "response" : """The Indian legal system addresses cases of online harassment and cyberbullying through various laws and regulations. The Information Technology Act, 2000, and its subsequent amendments provide legal provisions to tackle cybercrimes, including online harassment and cyberbullying. Additionally, the Indian Penal Code contains sections related to offenses such as defamation, stalking, and harassment, which can be applied to online behavior. Law enforcement agencies investigate such cases, and offenders can face penalties under thes elaws. \nReferences:\n- Information Technology Act, 2000\n- Indian Penal Code"""
        }
    else:
        return {"message": "Auth failed!"}


@app.get("/CHECK", status_code=200)
async def read_root():
    return {
        "connection": True,
        "deviceName": deviceName, 
        "message": "Connection successful!"}


if __name__ == '__main__':
    uvicorn.run(app, port=8001, host='192.168.1.6')
