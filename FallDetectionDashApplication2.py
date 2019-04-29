#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 12:24:33 2019

Purpose: This is a script to create the dash application for Activity detection. 

@author: zeski
"""

import os
import pickle

import pandas as pd
import numpy as np

print(os.listdir())

df = pd.read_csv("PrincipleComponent.csv")





### The code for prediction
sample = df.iloc[:, :-1].sample().values

print(sample)

XGB_model = pickle.load(open("XGB_component_model.pickle", "rb"))
prediction_encoder = pickle.load(open("EncodedLabels/Activity_encoder.pickle", 'rb'))

prediction = XGB_model.predict(sample)

print("Prediction: {}".format(prediction_encoder.inverse_transform(prediction)[0]))



### Dash Application code: 








