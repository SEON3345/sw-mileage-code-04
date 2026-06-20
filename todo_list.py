import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyDetectionSystem:
    def __init__(self, output_dir="outputs/anomaly"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=200,
            contamination=0.04,
            random_state=42
        )

    def create_sensor_like_data(self, size=800):
        np.random.seed(42)

        timestamp = pd.date_range(
            start="2025-01-01",
            periods=size,
            freq="10min"
        )

        x1 = 50 + 10 * np.sin(np.linspace(0, 20 * np.pi, size)) + np.random.normal(0, 2, size)
        x2 = 100 + 15 * np.cos(np.linspace(0, 12 * np.pi, size)) + np.random.normal(0, 3, size)
        x3 = 30 + np.random.normal(0, 4, size)

        anomaly_indices = np.random.choice(size, 30, replace=False)

        x1[anomaly_indices] += np.random.normal(35, 8, len(anomaly_indices))
        x2[anomaly_indices] -= np.random.normal(40, 10, len(anomaly_indices))
        x3[anomaly_indices] += np.random.normal(25, 6, len(anomaly_indices))

        df = pd.DataFrame({
            "timestamp": timestamp,
            "sensor_1": x1,
            "sensor_2": x2,
            "sensor_3": x3
        })

        return df

    def create_features(self, df):
        data = df.copy()

        for col in ["sensor_1", "sensor_2", "sensor_3"]:
            data[f"{col}_diff"] = data[col].diff().fillna(0)
            data[f"{col}_rolling_mean"] = data[col].rolling(6, min_periods=1).mean()
            data[f"{col}_rolling_std"] = data[col].rolling(6, min_periods=2).std().fillna(0)

        data["hour"] = data["timestamp"].dt.hour
        data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
        data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)

        return data

    def detect(self, data):
        feature_columns = [
            col for col in data.columns
            if col != "timestamp"
        ]

        X = data[feature_columns]
        X_scaled = self.scaler.fit_transform(X)

        prediction = self.model.fit_predict(X_scaled)
        score = self.model.decision_function(X_scaled)

        result = data.copy()
        result["anomaly_score"] = score
        result["is_anomaly"] = np.where(prediction == -1, 1, 0)

        return result

    def save_result(self, result):
        result.to_csv(
            self.output_dir / "anomaly_detection_result.csv",
            index=False,
            encoding="utf-8-sig"
        )

        anomalies = result[result["is_anomaly"] == 1]
        anomalies.to_csv(
            self.output_dir / "detected_anomalies.csv",
            index=False,
            encoding="utf-8-sig"
        )

        print("전체 데이터 수:", len(result))
        print("탐지된 이상치 수:", len(anomalies))

    def plot_result(self, result):
        normal = result[result["is_anomaly"] == 0]
        anomaly = result[result["is_anomaly"] == 1]

        plt.figure(figsize=(12, 5))
        plt.plot(normal["timestamp"], normal["sensor_1"], label="Normal", alpha=0.7)
        plt.scatter(
            anomaly["timestamp"],
            anomaly["sensor_1"],
            label="Anomaly",
            marker="x"
        )

        plt.title("Anomaly Detection Result")
        plt.xlabel("Timestamp")
        plt.ylabel("Sensor Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "anomaly_plot.png", dpi=150)
        plt.close()

    def run(self):
        df = self.create_sensor_like_data()
        data = self.create_features(df)
        result = self.detect(data)

        self.save_result(result)
        self.plot_result(result)

        print("이상탐지 완료")


if __name__ == "__main__":
    detector = AnomalyDetectionSystem()
    detector.run()
