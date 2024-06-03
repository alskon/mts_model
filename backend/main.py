from fastapi import FastAPI, UploadFile, status
from fastapi.responses import RedirectResponse, FileResponse
import datetime
import os
import json
import src.preprocessing as preprocessing
import src.prediction as prediction
import src.delete_files as del_f

app = FastAPI()
feature_importance, pred_proba = None, None
del_f.del_files('input')
del_f.del_files('output')


@app.get("/")
async def root():
    return {"message": "Backend MTS MLOPS_2"}

@app.get("/upload")
async def upload_file_get():
    return {"message": "backend /upload"}

@app.post("/upload")
async def upload_file_post(file: UploadFile):
    global pred_proba
    new_filename = f'{file.filename.split(".")[0]}_{"_".join(str(datetime.datetime.now())[:10].split(" "))}.csv'
    save_location = os.path.join('input', new_filename)
    with open(save_location, "wb") as f:
        f.write(file.file.read())

    preprocessed_df = preprocessing.main_preprocessing(save_location)

    submission, feature_importance, pred_proba = prediction.prediction(preprocessed_df)
    submission.to_csv(save_location.replace('input', 'output'), index=False)

    with open(os.path.join('output', 'json_file.json'), 'a') as file:
        file.write(json.dumps(feature_importance, indent=4, ensure_ascii=False))

    return RedirectResponse(url='/download', status_code=status.HTTP_303_SEE_OTHER)

@app.get("/download")
async def download_file_get():
    return json.dumps({
        'files': [el for el in os.listdir('output')],
        'hist_data': pred_proba.tolist()
    })

@app.get("/download_file/{filename}")
async def download_file(filename):
    return FileResponse(f'./output/{filename}', filename=filename)




