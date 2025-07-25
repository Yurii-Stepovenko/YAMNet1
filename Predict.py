import tensorflow as tf
import numpy as np
import sounddevice as sd
import queue
import joblib
import tensorflow_hub as hub
from collections import Counter

#Функція для зчитування з мікрофону
def predict_realtime_rf(
    yamnet_model,
    rf_model,
    class_names={0: "background", 1: "motorcycle"},
    segment_duration=0.96
):
    sample_rate = 16000
    segment_samples = int(segment_duration * sample_rate)
    q = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"{status}")
        q.put(indata.copy())

    print("Слухаю... (CTRL+C для зупинки)")

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype='float32',
            blocksize=segment_samples,
            callback=audio_callback
        ):
            while True:
                audio_chunk = q.get()
                wav = np.squeeze(audio_chunk)
                if wav.ndim != 1:
                    wav = wav[:, 0]

                segment_tensor = tf.convert_to_tensor(wav, dtype=tf.float32)
                _, embeddings, _ = yamnet_model(segment_tensor)
                mean_embedding = tf.reduce_mean(embeddings, axis=0).numpy().reshape(1, -1)

                pred_class = rf_model.predict(mean_embedding)[0]
                prob = rf_model.predict_proba(mean_embedding)[0][pred_class]

                print(f"➡️ {class_names[pred_class]} (ймовірність: {prob:.2f})")

    except KeyboardInterrupt:
        print("\n Зупинено користувачем.")
    except Exception as e:
        print(f"Помилка: {e}")


#Основний блок
if __name__ == "__main__":
    #Завантаження моделі YAMNet
    print("Завантаження YAMNet...")
    yamnet_model = hub.load('D:\YAMNet1\model\chive')

    #Завантаження Random Forest моделі
    print("Завантаження Random Forest моделі...")
    rf_model = joblib.load("model_rf.pkl")

    #Запуск
    predict_realtime_rf(yamnet_model, rf_model)
