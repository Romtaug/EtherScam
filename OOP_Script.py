import os
import shutil
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pymongo import MongoClient
import joblib
from xgboost import XGBClassifier
import requests

class EtherscanScraper:
    def __init__(self, base_url, max_pages):
        self.base_url = base_url
        self.max_pages = max_pages
        self.download_dir = os.getcwd()
        self.data_dir = os.path.join(self.download_dir, "data_etherscan")
        os.makedirs(self.data_dir, exist_ok=True)
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.csv_files = []
        self.driver = self.init_driver()

    def init_driver(self):
        options = Options()
        options.use_chromium = True
        prefs = {"download.default_directory": self.download_dir}
        options.add_experimental_option("prefs", prefs)
        return webdriver.Edge(options=options)

    def get_latest_downloaded_file(self):
        downloaded_files = sorted(
            [f for f in os.listdir(self.download_dir) if f.endswith(".csv")],
            key=lambda x: os.path.getctime(os.path.join(self.download_dir, x)),
            reverse=True
        )
        return downloaded_files[0] if downloaded_files else None

    def extract_transactions(self):
        for page in range(1, self.max_pages + 1):
            url = self.base_url + str(page)
            self.driver.get(url)
            WebDriverWait(self.driver, 15).until(
                EC.element_to_be_clickable((By.ID, "btnExportQuickTransactionListCSV"))
            ).click()
            time.sleep(5)

            latest_file = self.get_latest_downloaded_file()
            if latest_file:
                src_path = os.path.join(self.download_dir, latest_file)
                dst_path = os.path.join(self.data_dir, f"page_{page}_{self.timestamp}.csv")
                shutil.move(src_path, dst_path)
                self.csv_files.append(dst_path)
        self.driver.quit()

class FileMerger:
    def __init__(self, data_dir, timestamp):
        self.data_dir = data_dir
        self.timestamp = timestamp
        self.df_merged = pd.DataFrame()

    def merge_csv(self):
        csv_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith(".csv")]
        dfs = [pd.read_csv(f) for f in csv_files]
        self.df_merged = pd.concat(dfs).drop_duplicates(subset='Txn Hash')
        return True

    def save_merged_file(self):
        export_dir = os.path.join(os.getcwd(), "mongo_etherscan")
        os.makedirs(export_dir, exist_ok=True)
        path = os.path.join(export_dir, f"etherscan_transactions_full_{self.timestamp}.csv")
        self.df_merged.to_csv(path, index=False)

class DatasetImporter:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.local_dir = os.path.join(os.getcwd(), "data_kaggle")
        os.makedirs(self.local_dir, exist_ok=True)
        self.csv_file = None

    def locate_csv(self):
        for file in os.listdir(self.dataset_path):
            if file.endswith(".csv"):
                src = os.path.join(self.dataset_path, file)
                dst = os.path.join(self.local_dir, file)
                shutil.copy(src, dst)
                self.csv_file = dst
                break

    def load_dataframe(self):
        return pd.read_csv(self.csv_file) if self.csv_file else None

class DatabaseManager:
    def __init__(self):
        self.client = MongoClient("mongodb://localhost:27017")
        self.db = self.client["fraud_detection"]
        self.collection = self.db["transactions"]

    def store_data(self, df):
        self.collection.drop()
        self.collection.insert_many(df.to_dict(orient="records"))

    def retrieve_data(self):
        df = pd.DataFrame(list(self.collection.find()))
        return df.drop(columns=["_id"], errors='ignore')

class DatasetCleaner:
    def __init__(self, df):
        self.df = df

    def clean(self):
        self.df.columns = self.df.columns.str.strip()
        self.df.drop(columns=['Address'], errors='ignore', inplace=True)
        self.df.drop_duplicates(inplace=True)
        self.df.dropna(inplace=True)
        return self.df

class ResultExporter:
    def __init__(self, df, filename):
        self.df = df
        self.filename = filename

    def save_without_flag(self):
        self.df.drop(columns=["FLAG"], errors='ignore').to_csv(self.filename, index=False)

class FraudClassifier:
    def __init__(self, df):
        self.df = df
        self.model = XGBClassifier()
        self.best_params = {}
        self.best_accuracy = 0
        self.results = []

    def preprocess(self):
        cleaner = DatasetCleaner(self.df)
        self.df = cleaner.clean()

    def train(self):
        X = self.df.drop(columns=['FLAG'])
        y = self.df['FLAG']
        self.model.fit(X, y)

    def save_model(self, path):
        joblib.dump(self.model, path)

    def load_model(self, path):
        self.model = joblib.load(path)

    def evaluate_wallet(self, address, api_key):
        url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&apikey={api_key}"
        df = pd.DataFrame(requests.get(url).json()["result"])
        features = {...}  # À implémenter : extraction des features depuis df
        prediction = self.model.predict(pd.DataFrame([features]))[0]
        probability = self.model.predict_proba(pd.DataFrame([features]))[0][1]
        return prediction, probability