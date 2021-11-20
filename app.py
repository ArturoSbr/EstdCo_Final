# Libraries
from flask import Flask, jsonify, request
import pickle
import numpy as np

# Server
app = Flask(__name__)

# Prueba
@app.route('/interfaz/', methods=['GET'])
def ping():
    return jsonify({'message':'pong'})

# Predict requested income(s)
@app.route('/predict/', methods=['POST'])
def predict():
    # Load model
    m = pickle.load(open('model/gbr.pkl', 'rb'))
    # Request to numpy
    X = np.array([list(x.values()) for x in request.json])
    # Predict
    y_hat = list(m.predict(X))
    # Return json
    return jsonify(y_hat)

# Recalibrar modelo
@app.route('/recalibrate/', methods=['POST'])
def recalibrate():
    # Targets
    y = np.array([d['ingreso'] for d in request.json])
    print('\nA columna de target:\n', y)
    # Features
    X = np.array([list(d.values())[1:] for d in request.json])
    print('\nA columnas de features:\n', X)
    # hola
    # Return success message
    return jsonify({'message':'recibí solicitud'})
    

# Correr app cuando se ejecuta el script
if __name__ == '__main__':
    app.run(debug=True, port=4000)
