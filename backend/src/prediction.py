from catboost import CatBoostClassifier
import pandas as pd
import numpy as np

model_path = './/model//model_cb.cbm'
THRESHOLD = 0.505


def prediction(input_df):
    model = CatBoostClassifier()
    model.load_model(model_path)
    feature_importance = {el[0]: el[1] for el in zip(input_df.columns, model.get_feature_importance())}
    sort_feature_importance = sorted(feature_importance.items(), key=lambda item: item[1])[-5:]
    top_5_feature_importance = {el[0]: el[1] for el in sort_feature_importance}

    preds = model.predict_proba(input_df)[:, 1]

    result = pd.DataFrame({
        'client_id': input_df.index,
        'preds': np.int8(preds > THRESHOLD)
    })
    return result, top_5_feature_importance, preds
