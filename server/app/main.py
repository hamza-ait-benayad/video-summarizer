from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
import uvicorn
from app.database import engine
from app.models import base
from app.api.routes import video

base.Base.metadata.create_all(bind=engine)

app = FastAPI()



app.include_router(video.router)

@app.get("/")
def read_root():
    return {"Hello": "World"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8004, reload=True)
