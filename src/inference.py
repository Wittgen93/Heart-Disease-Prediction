import joblib
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from model_training import Net  # или откуда вы импор­тируете Net
import argparse

def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=['output'])
    return X, df

def main(args):
    X, raw = load_data(args.input)
    num_cols = ['age','trtbps','chol','thalachh','oldpeak']
    scaler = joblib.load('models/scaler.pkl')
    X[num_cols] = scaler.transform(X[num_cols])

    lr = joblib.load('models/lr_model.pkl')
    rf = joblib.load('models/rf_model.pkl')

    probs = {
        'lr': lr.predict_proba(X)[:,1],
        'rf': rf.predict_proba(X)[:,1]
    }

    X_tensor = torch.tensor(X.values.astype('float32'))
    dnn = Net(X_tensor.shape[1])
    dnn.load_state_dict(torch.load('models/dnn_model.pt'))
    dnn.eval()
    with torch.no_grad():
        probs['nn'] = dnn(X_tensor).numpy().ravel()

    out = pd.DataFrame(probs, index=raw.index)
    out.to_csv(args.output, index=True)
    print(f"Saved predictions to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  type=str, default='data/processed/heart_preprocessed.csv')
    parser.add_argument('--output', type=str, default='predictions.csv')
    args = parser.parse_args()
    main(args)