from flask import Flask, request
import pickle

model_mlp_adam = pickle.load(open('model_mlp_adam.pkl', 'rb'))
model_mlp_lbfgs = pickle.load(open('model_mlp_lbfgs.pkl', 'rb'))
model_mlp_sgd = pickle.load(open('model_mlp_sgd.pkl', 'rb'))

model_decisiontree = pickle.load(open('model_decisiontree.pkl', 'rb'))
model_randomforest = pickle.load(open('model_randomforest.pkl', 'rb'))

model_xgb = pickle.load(open('model_xgb.pkl', 'rb'))

app = Flask(__name__)

def model(model_name):
    
    step = int(request.args.get('step', ''))
    amount = float(request.args.get('amount', ''))
    
    oldbalanceOrg = float(request.args.get('oldbalanceOrg', ''))
    newbalanceOrig = float(request.args.get('newbalanceOrig', ''))
    
    oldbalanceDest = float(request.args.get('oldbalanceDest', ''))
    newbalanceDest = float(request.args.get('newbalanceDest', ''))
    
    errorbalanceOrg = float(request.args.get('errorbalanceOrg', ''))
    errorbalanceDest = float(request.args.get('errorbalanceDest', ''))
    
    type_CASH_OUT = int(request.args.get('type_CASH_OUT', ''))
    type_TRANSFER = int(request.args.get('type_TRANSFER', ''))
    
    predictions = ''
    if model_name == 'home':
        model_name = [model_mlp_adam, model_mlp_lbfgs, model_mlp_sgd, model_randomforest, model_decisiontree, model_xgb]
        for name in model_name:
            prediction = name.predict([[
                                    step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, \
                                    errorbalanceOrg, errorbalanceDest, type_CASH_OUT, type_TRANSFER
                                    ]])
            
            predictions += str(prediction)
            print(predictions)
        return predictions
        
    elif model_name == 'mlp':
        model_name = [model_mlp_adam, model_mlp_lbfgs, model_mlp_sgd]
        for name in model_name:
            prediction = name.predict([[
                                    step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, \
                                    errorbalanceOrg, errorbalanceDest, type_CASH_OUT, type_TRANSFER
                                    ]])
            predictions += str(prediction)
            print(predictions)
        return predictions

    else:
        prediction = model_name.predict([[
        step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, \
        errorbalanceOrg, errorbalanceDest, type_CASH_OUT, type_TRANSFER
        ]])
        
        predictions += str(prediction)
        print(predictions)
        return predictions

@app.route('/predict', methods=['POST'])
def home():
    return model('home')

@app.route('/predict/mlp', methods=['POST'])
def mlp():
    return model('mlp')
    
@app.route('/predict/mlp/adam', methods=['POST'])
def mlp_adam():
    return model(model_mlp_adam)

@app.route('/predict/mlp/lbfgs', methods=['POST'])
def mlp_lbfgs():
    return model(model_mlp_lbfgs)

@app.route('/predict/mlp/sgd', methods=['POST'])
def mlp_sgd():
    return model(model_mlp_lbfgs)

@app.route('/predict/randomforest', methods=['POST'])
def randomforest():
    return model(model_randomforest)

@app.route('/predict/decisiontree', methods=['POST'])
def decisiontree():
    return model(model_decisiontree)

@app.route('/predict/xgb', methods=['POST'])
def xgb():
    return model(model_xgb)

app.run(debug=True)
