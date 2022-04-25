import os 
import shutil 
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from utils import TrainingVisualizationCb
import numpy as np

class LSTM_Model:
    
    def __init__(self, model_name, input_dim, settings):
        self.model_name = model_name
        self.settings = settings
        
        if os.path.exists(f"./models/{model_name}"):
            self.model = keras.models.load_model(f"./models/{model_name}/model.h5")
            return
        
        os.mkdir(f"./models/{model_name}")
        shutil.copyfile("model_settings.py", f"./models/{model_name}/mod_settings.py")
        
         # Define and compile LSTM model
        self.model = Sequential()
        self.model.add(LSTM(100, return_sequences=True, input_shape=(settings.SEQUENCE_LENGTH, input_dim)))
        self.model.add(Dropout(settings.DROPOUT_RATE))
        self.model.add(LSTM(50, return_sequences=True))
        self.model.add(Dropout(settings.DROPOUT_RATE))
        self.model.add(LSTM(25, return_sequences=False))
        self.model.add(Dropout(settings.DROPOUT_RATE))
        #model.add(LSTM(50))
        #model.add(Dropout(0.2))
        self.model.add(Dense(1, activation="tanh"))
        opt = tf.keras.optimizers.Adam(learning_rate=settings.LR)
        self.model.compile(optimizer=opt, loss='mse')#,  metrics=[''])
        
        # Write model summary (architecture) to file
        with open(f"./models/{model_name}/model_summary.txt","w") as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))
        
        
    def fit(self, X_train, y_train, X_valid, y_valid):
        # Callbacks for training
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='max', verbose=2, patience=5)
        mc = tf.keras.callbacks.ModelCheckpoint(f'./models/{self.model_name}/model.h5', mode='min', verbose=2, save_best_only=True)
        viz = TrainingVisualizationCb()
        
        # Train the model 
        history = self.model.fit(X_train, y_train,
                            validation_data=(X_valid, y_valid),
                            epochs=20, verbose=2, callbacks=[mc, viz])
    
    def forecast(self, X, start_ind, forecast_window_len=24):
        seq_len = self.settings.SEQUENCE_LENGTH
        
        #start_ind = np.random.choice(X.shape[0] - seq_len - 1)
        model_input = X[start_ind : start_ind + seq_len]
        forecasts = []

        for pred_no in range(1, forecast_window_len + 1):
            forecast = self.model(np.array([model_input]))
            forecasts.append(forecast[0][0])
            model_input = X[start_ind + pred_no : start_ind + pred_no + seq_len]
            model_input[-pred_no:, -1] = np.array(forecasts)
        return np.array(forecasts)