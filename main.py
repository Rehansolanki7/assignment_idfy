from fastapi import FastAPI, UploadFile, File
from prediction import prediction
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
import uvicorn
import tempfile

app = FastAPI()
 
@app.post("/form_classification/predict")
async def form_classification(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(await file.read())
        output = prediction(tmp_file.name)
        output = {"Prediction":output}
    except Exception as e:
        output = {"Error":str(e)}
    headers = {"Access-Control-Allow-Origin": "*"}
    return JSONResponse(content=output, headers=headers)

if __name__ == "__main__":
    uvicorn.run(app)