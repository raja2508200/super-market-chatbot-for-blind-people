from flask import Flask, render_template, request, flash, redirect, url_for, session
import sqlite3
import random
import numpy as np
import csv
import pickle
import json
from flask_ngrok import run_with_ngrok
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import speech_recognition as sr
from gtts import gTTS
import os
from googletrans import Translator
import re
app = Flask(__name__)


lemmatizer = WordNetLemmatizer()

model = load_model("chatbot_mode2.h5")
intents = json.loads(open("intents2.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))



def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    result = "Sorry, I don't understand that."

    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break

    return result

def chatbot_response(msg):
    messg = msg.lower()
    print(msg)
    
    ints = predict_class(msg, model)
    print(ints[0]["intent"])
    res = getResponse(ints, intents)
    return res


    
    # chat functionalities
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

@app.route('/')
def index():
    return render_template('new.html')

products={"tomato":15,"onion":40,"potato":30,"carrot":25,"brinjal":40,"cucumber":35,"apple":50,"banana":35,"orange":45,"mango":60,"pineapple":55,"grapes":70,"chicken":120,"mutton":300,"beef":250,"pork":180,"milk":50,"yogurt":40,"cheese":200,"butter":120}
product={}
totall={}
order=[]
quan=[]
button=[0]
@app.route('/start')
def start():
    speech = speak1()
    if speech=="false":
            output="cannot understand your voice please tell me again"
            response_audio_filename = f"response_{os.urandom(8).hex()}.mp3"
            tts = gTTS(text=output, lang='en')
            tts.save(os.path.join("static", response_audio_filename))
            if button[-1]==3:
                if len(quan)==len(order):
                        j=generatebill()
                        response_audio_filename = f"response_{os.urandom(8).hex()}.mp3"
                        tts = gTTS(text=output, lang='en')
                        tts.save(os.path.join("static", response_audio_filename))
                        button.append(3)
                        return render_template('output.html',audio_file=response_audio_filename,l=totall,j=j)
                else:
                        return render_template('new.html',audio_file=response_audio_filename)

    b=speech.split()
    print(b,"b")
    p=b[-1]
    print(p)
    if button[-1]==0:
            print(p)
            if p not in products:
                output=chatbot_response(speech)
                if output=="bill":
                    if len(quan)==len(order):
                        j=generatebill()
                    print(j)
                    print(product)
                    output=f"your products {totall} finally your amont {j}rupees if you want to continue shoping or recepit"
                    response_audio_filename = f"response_{os.urandom(8).hex()}.mp3"
                    tts = gTTS(text=output, lang='en')
                    tts.save(os.path.join("static", response_audio_filename))
                    button.append(0)
                    return render_template('new.html',audio_file=response_audio_filename)
                elif output=="remove":
                    if len(order)>0:
                        output="tell me your product name"
                        response_audio_filename = f"response_{os.urandom(8).hex()}.mp3"
                        tts = gTTS(text=output, lang='en')
                        tts.save(os.path.join("static", response_audio_filename))
                        button.append(2)
                        return render_template('new.html',audio_file=response_audio_filename)
                elif output=="receipt":
                    if len(order)>0:
                        print("YES")
                        if len(quan)==len(order):
                            j=generatebill()
                        output=f"your products {totall} finally your amont {j}rupees, tell me your mobile number"
                        response_audio_filename = f"response_{os.urandom(8).hex()}.mp3"
                        tts = gTTS(text=output, lang='en')
                        tts.save(os.path.join("static", response_audio_filename))
                        button.append(3)
                        return render_template('output.html',audio_file=response_audio_filename,l=totall,j=j)
                    else:
                        output="no products available please purchase something "
                        response_audio_filename = f"response_{os.urandom(8).hex()}.mp3"
                        tts = gTTS(text=output, lang='en')
                        tts.save(os.path.join("static", response_audio_filename))
                        button.append(0)
                        return render_template('new.html',audio_file=response_audio_filename)
                elif output=="add":
                        output="tell me your product name"
                        response_audio_filename = f"response_{os.urandom(8).hex()}.mp3"
                        tts = gTTS(text=output, lang='en')
                        tts.save(os.path.join("static", response_audio_filename))
                        button.append(0)
                        return render_template('new.html',audio_file=response_audio_filename)
                else:
                    response_audio_filename = "input.mp3"
                    tts = gTTS(text=output, lang='en')
                    tts.save(os.path.join("static", response_audio_filename))
                    button.append(0)
                    return render_template('new.html',audio_file=response_audio_filename)
            else:
                output="how many kilo grams you want"
                response_audio_filename = f"response_{os.urandom(8).hex()}.mp3"
                tts = gTTS(text=output, lang='en')
                tts.save(os.path.join("static", response_audio_filename))
                button.append(1)
                order.append(p)
                return render_template('new.html',audio_file=response_audio_filename)
    elif button[-1]==1:
        try:
            q=int(speech[0])
            quan.append(q)
            output="successfully add your product now you can add more products"
            response_audio_filename = f"response_{os.urandom(8).hex()}.mp3"
            tts = gTTS(text=output, lang='en')
            tts.save(os.path.join("static", response_audio_filename))
            button.append(0)
            return render_template('new.html',audio_file=response_audio_filename)
        except:
            output="invalid kilograms please tell me again"
            response_audio_filename = f"response_{os.urandom(8).hex()}.mp3"
            tts = gTTS(text=output, lang='en')
            tts.save(os.path.join("static", response_audio_filename))
            return render_template('new.html',audio_file=response_audio_filename)
    elif button[-1]==2:
          if len(quan)==len(order):
              for i in b:
                     if i in order:
                        for l,m in enumerate(order):
                            if m==i:
                                order.remove(m)
                                quan.pop(l)
              j=generatebill()
              output=f"successfully remove .now your products {totall} finally your amont {j}rupees"
              response_audio_filename = f"response_{os.urandom(8).hex()}.mp3"
              tts = gTTS(text=output, lang='en')
              tts.save(os.path.join("static", response_audio_filename))
              button.append(0)
              return render_template('new.html',audio_file=response_audio_filename)
    elif button[-1]==3:
        if is_valid_mobile_number(speech):
                print(speech)
                speech1=str(speech)
                k_without_spaces = speech1.replace(" ", "")
                table_name = "custemer_id" + k_without_spaces
                conn = sqlite3.connect('items.db')
                cursor = conn.cursor()
                cursor.execute("CREATE TABLE IF NOT EXISTS {} (item TEXT, quantity INTEGER, unit TEXT, price INTEGER, currency TEXT)".format(table_name))
                for item, details in totall.items():
                    cursor.execute(f"INSERT INTO {table_name} (item, quantity, unit, price, currency) VALUES (?, ?, ?, ?, ?)",
                                   (item, details[0], details[1], details[2], details[3]))
                    conn.commit()
                output="Thank you for visiting our supermarket! Please come again. You're always welcome!"
                response_audio_filename = f"response_{os.urandom(8).hex()}.mp3"
                tts = gTTS(text=output, lang='en')
                tts.save(os.path.join("static", response_audio_filename))
                button.append(0)
                order.clear()
                quan.clear()
                product.clear()
                totall.clear()
                return render_template('new.html',audio_file=response_audio_filename)
        else:
            if len(quan)==len(order):
                j=generatebill()
                output="invalid mobile number please tell me again"
                response_audio_filename = f"response_{os.urandom(8).hex()}.mp3"
                tts = gTTS(text=output, lang='en')
                tts.save(os.path.join("static", response_audio_filename))
                button.append(3)
                return render_template('output.html',audio_file=response_audio_filename,l=totall,j=j)

            
    else:
            return "invalid orders"

        
            

        
        
        
def is_valid_mobile_number(mobile_number):
    mobile_number = re.sub(r'\D', '', mobile_number)
    mobile_number_regex = r'^\d{10}$'
    return re.match(mobile_number_regex, mobile_number) is not None

r = sr.Recognizer()
mic = sr.Microphone()
def speak1():
    try:
        with mic as audio_file:
            print("Speak Now...")
            r.adjust_for_ambient_noise(audio_file)
            audio = r.listen(audio_file)
            print("Converting Speech to Text...")
            text = r.recognize_google(audio)
            text = text.lower()
            print("Input:", text)
            return text
    except:
        return "false"
   
        
def generatebill():
    for i in range(len(quan)):
        vegita=order[i]
        quan1=quan[i]
        product[vegita]=products[vegita]*quan1
        totall[vegita]=[quan1,"kg",int(products[vegita]*quan1),"rupees"]
        j=sum(product.values())
    return j




if __name__ == '__main__':
    app.run(port=600)   
