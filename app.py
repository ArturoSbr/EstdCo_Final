# Importar librerías
from flask import Flask, jsonify, request
import pickle
import numpy as np

# Servidor
app = Flask(__name__)

# Probar si el servidor responde
@app.route('/ping/', methods=['GET'])
def ping():
    return jsonify({'message':'Recibí un ping. Contesto con un pong.'})

# Predecir ingresos solicitados
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

# Recibir nuevos registros, agregarlos a SQL y recalibrar el modelo
@app.route('/recalibrate/', methods=['POST'])
def recalibrate():
    # Cargar solicitud como lista de listas
    new = [list(d.values()) for d in request.json]
    print(new)
    # # Subir cada registro a SQL
    # for reg in new:
    #     sql.upload(reg)
    # Return success message
    return jsonify(
        {
            'message':'Recibí nuevos registros.',
            'to-do':'Subir los nuevos registros a SQL, bajar todos los registros a memoria y volver a ajustar el modelo.'
        }
    )
    

# Correr app cuando se ejecuta el script
if __name__ == '__main__':
    app.run(debug=True, port=4000)
