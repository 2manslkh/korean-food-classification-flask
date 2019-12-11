import json
import model.predict
import base64

from flask import Flask ,request

# create the flask object
app = Flask(__name__)

@app.route('/predict',methods=['GET','POST'])
def predict():

    # print(request.data)
    # print(request.form)
    img_bytes = request.form.get('img_bytes')

    if img_bytes == None:
        print('got none')
        return 'u stupid'

    # print(img_bytes[:10])
    img_bytes = base64.b64decode(img_bytes)
    return json.dumps(str(model.predict.predict(img_bytes)))

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)
