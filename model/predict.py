from tensorflow import keras, image, float32,math,reshape
from tensorflow_hub import KerasLayer
import numpy as np
import json

model = keras.models.load_model('model/mobileNetV2_final_hub.h5',custom_objects={'KerasLayer':KerasLayer})
# model = keras.models.load_model('model/my_model_Mn2.h5')

d = {0: 'samgyeopsal', 1: 'bulgogi', 2: 'ojingeo_bokkeum', 3: 'dakbokkeumtang', 4: 'galchijorim', 5: 'jeyuk_bokkeum', 6: 'ramyeon', 7: 'bibimbap', 8: 'galbijjim', 9: 'kimchi'}

def predict(img_bytes):

    # Decode
    x = image.decode_jpeg(img_bytes, channels=3)
    print('decoded')

    # Normalize
    x = image.convert_image_dtype(x, float32)
    print('normalized')

    # Resize
    np_x = image.resize(x, [224, 224]).numpy()
    x = np_x.tolist()
    print('resized')

    # Predict
    pred = model.predict(reshape(x,[-1,224,224,3]))

    # Map
    top_k_values, top_k_indices = math.top_k(
        pred,
        k=3,
        sorted=True,
        name=None
    )

    top_k_values = top_k_values.numpy().tolist()[0]
    top_k_labels = [d[idx] for idx in top_k_indices.numpy()[0]]
    results = {'prob':top_k_values,'prob_labels':top_k_labels}
    print(results)

    # Return
    return json.dumps(results)
    