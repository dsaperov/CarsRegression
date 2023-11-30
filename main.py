import os
import pickle

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

with open(os.path.join('pickles', 'model.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join('pickles', 'encoder.pkl'), 'rb') as f:
    encoder = pickle.load(f)

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


def modify_types(df):
    extraction_regex = r'(\d+(?:\.\d)?\d*)'

    df['mileage'] = df['mileage'].str.extract(extraction_regex).astype(float)
    df['engine'] = df['engine'].str.extract(extraction_regex).astype(int)
    df['max_power'] = df['max_power'].str.extract(extraction_regex).astype(float)
    df['seats'] = df['seats'].astype(int).astype(object)

    return df


def ohencode(df):
    df_obj = df.select_dtypes(include='object')
    df_num = df.select_dtypes(exclude='object')

    df_obj_encoded_np = encoder.transform(df_obj)
    df_obj_encoded = pd.DataFrame(df_obj_encoded_np.toarray(), columns=encoder.get_feature_names_out())
    df_encoded = pd.concat([df_obj_encoded, df_num], axis=1)

    return df_encoded


def modify_features(df):
    df['km_driven'] = np.log(df['km_driven'])
    df['year'] = df['year'] ** 2
    return df


def process_df(df):
    df.drop(['torque', 'name'], axis=1, inplace=True)
    df_types_modified = modify_types(df)
    df_encoded = ohencode(df_types_modified)
    df_features_modified = modify_features(df_encoded)
    return df_features_modified


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    item_df = process_df(pd.DataFrame([item.model_dump()]))
    prediction = model.predict(item_df)
    return prediction


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    items_df = process_df(pd.DataFrame([item.model_dump() for item in items]))
    predictions = model.predict(items_df)
    return predictions.tolist()