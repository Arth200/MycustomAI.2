import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import yaml
import os

class MyCustomAI:
    def __init__(self, config_path="config/config.yaml"):
        """
        Initialise le modèle MyCustomAI en chargeant la configuration depuis un fichier YAML.

        Args:
            config_path (str): Chemin vers le fichier de configuration YAML.
        """
        self.load_config(config_path)
        self.scaler = StandardScaler()
        self.model = MLPClassifier(**self.config['model_params'])
    
    def load_config(self, config_path):
        """
        Charger la configuration depuis un fichier YAML.

        Args:
            config_path (str): Chemin du fichier de configuration.
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
    
    def preprocess_data(self, X):
        """
        Prétraiter les données en les standardisant.

        Args:
            X (array-like): Les données d'entrée à prétraiter.

        Returns:
            array-like: Les données prétraitées.
        """
        return self.scaler.transform(X)
    
    def train(self, X, y):
        """
        Entraîner le modèle avec des données.

        Args:
            X (array-like): Caractéristiques d'entrée.
            y (array-like): Étiquettes de sortie.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, **self.config['train_test_split_params'])
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"Accuracy: {accuracy * 100:.2f}%")
    
    def predict(self, X):
        """
        Prédire les résultats pour de nouvelles données.

        Args:
            X (array-like): Les nouvelles données d'entrée.

        Returns:
            array-like: Les prédictions du modèle.
        """
        X_processed = self.preprocess_data(X)
        return self.model.predict(X_processed)
    
    def save_model(self, file_path):
        """
        Sauvegarder le modèle dans un fichier.

        Args:
            file_path (str): Chemin du fichier pour sauvegarder le modèle.
        """
        from joblib import dump
        dump((self.scaler, self.model), file_path)
        print(f"Model saved to {file_path}")
    
    def load_model(self, file_path):
        """
        Charger le modèle à partir d'un fichier.

        Args:
            file_path (str): Chemin du fichier d'où charger le modèle.
        """
        from joblib import load
        self.scaler, self.model = load(file_path)
        print(f"Model loaded from {file_path}")
