from logger import logtool
import uvicorn
import torch
from fastapi import FastAPI
from pydantic import BaseModel



class ConnectionItem(BaseModel):
    auth_token: str


class QueryItem(BaseModel):
    auth_token: str
    prompt: str


app = FastAPI()
deviceName = torch.cuda.get_device_name(0)

@app.post("/RAG")
async def RAG_request(query_item: QueryItem):
    if query_item.auth_token == "ISAUodiuIAU21":
        response = ""
        return {
            "prompt": query_item.prompt,
            "response" : response
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
    uvicorn.run(app, port=8001, host='192.168.1.4')
