import os
from math import floor
import io
import requests
from PIL import Image
from flask_wtf import FlaskForm
from flask import request
from flask import Flask
from wtforms import StringField, PasswordField, SubmitField, BooleanField, TextAreaField
from wtforms.validators import DataRequired, Length, Email,EqualTo, ValidationError
from flask import render_template, url_for, flash, redirect, request, abort
import numpy as np
import cv2
import csv
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
import pickle



tamilCharacterCode = []
model = None

app = Flask(__name__)



@app.route('/')
@app.route('/home')
def homepage_func():
	return render_template('homepage.html')

def bbox2(img1):
  img = 1 - img1
  rows = np.any(img, axis=1)
  cols = np.any(img, axis=0)
  rmin, rmax = np.where(rows)[0][[0, -1]]
  cmin, cmax = np.where(cols)[0][[0, -1]]
  return rmin, rmax, cmin, cmax
def RR(img):
    rmin, rmax, cmin, cmax = bbox2(img)
    # print(rmin, rmax, cmin, cmax)
    npArr = img[rmin:rmax, cmin:cmax]
    npArr = cv2.resize(npArr, dsize=(100, 100))
    jinga = np.ones((128,128))
    jinga[14:114,14:114] = npArr
    npArr = jinga.reshape(128, 128 , 1)
    return npArr

def getTamilChar(tamilCharacterCode, indx):
	return tamilCharacterCode[indx]

    
@app.route('/postmethod', methods = ['POST'])
def get_post_javascript_data():
	global tamilCharacterCode, model
	att = request.data

	imgStr = att.decode('utf-8')
	imgArr = imgStr.split(',')
	npArr = np.asarray(imgArr, dtype=np.uint8).reshape(400,400)
	
		
	npArr = RR(npArr)
	npArr = npArr.reshape(1, 128, 128 , 1)
	atc = model.predict(npArr)
	
	percentage = atc[0]

	valsss = atc[0].argsort()[-3:][::-1]
	
	responseTextSt = getTamilChar(tamilCharacterCode,valsss[0])+","+ getTamilChar(tamilCharacterCode,valsss[1])+ ","+ getTamilChar(tamilCharacterCode,valsss[2])
	
	responseTextSt = responseTextSt + ',%.3f,%.3f,%.3f'%(percentage[valsss[0]] *100.0,percentage[valsss[1]] *100.0,percentage[valsss[2]]*100.0)
	
	return responseTextSt


def init_somethings():

	global tamilCharacterCode, model

	with open('unicodeTamil.csv', newline='') as f:
		reader = csv.reader(f)
		data = list(reader)
		for i in data:
			go = i[1].split(' ')
			charL = ""
			for gg in go:
				charL = charL + "\\u"+str(gg)
			tamilCharacterCode.append(charL.encode('utf-8').decode('unicode-escape'))
	
	
	model = load_model('tamilALLEzhuthukalKeras_Model.h5')
	print(model.summary())

if __name__ == '__main__':
	init_somethings()
	#print(tamilCharacterCode)
	print("\n*******************App started*****************\n")
	# run overall
	# app.run(debug=True,  host='0.0.0.0')
	# run in localhost
	app.run(debug=True)
