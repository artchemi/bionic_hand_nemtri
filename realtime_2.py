"""
@original author: Amir Modan
@editor: Jimmy L. @ SF State MIC Lab
 - Date: Summer 2022

Main Program for Real-Time system which establishes BLE connection,
    defines GUI, and finetunes realtime samples from a pretrained finetune-base model.
    
Flow: (running via Async Functions)
    1. Run the code (python realtime.py)
    2. Enable bluetooth in setting, and code with automatically pair with armband
    3. Follow instructions to perform gestures for finetuning
    4. Finetune training starts
        - Optional: save finetuned-model
    5. Real time gesture recognition begins

Note:
    1. Should see myo armband in blue lighting if connected.
    2. Run this on Linux if possible, sometimes Bleak refuses to connect to Myo Armband under Windows environment.
"""

import asyncio
import json
import random
import time
import config
import nest_asyncio
nest_asyncio.apply()
import tensorflow as tf
import numpy as np
import warnings
from typing import Any
from bleak import BleakClient, BleakScanner, discover
from dataset import realtime_preprocessing
from model import get_finetune, realtime_pred


warnings.filterwarnings("ignore")
tf.get_logger().setLevel('INFO')

# UUID's for BLE Connection

CONTROL = "d5060401-a904-deb9-4748-2c7f4a124842"
EMG0 = "d5060105-a904-deb9-4748-2c7f4a124842"
EMG1 = "d5060205-a904-deb9-4748-2c7f4a124842"
EMG2 = "d5060305-a904-deb9-4748-2c7f4a124842"
EMG3 = "d5060405-a904-deb9-4748-2c7f4a124842"

# Batch size for realtime fine-tuning
realtime_batch_size = 2

# Epoch for realtime fine-tuning
realtime_epochs = 15

# Samples window
window = 32

# Step Size
step_size = 10

# Samples to be recored for each gesture
SAMPLES_PER_GESTURE = 10 * window

# List of Gestures to be used for classification

GESTURES = [
    "Rest", "Fist", "Thumbs Up", "Ok Sign"
]

delay = 2.0

# Number of sensors Myo Armband contains
num_sensors = 8

# Path to save finetuned model, set NONE if no export
# finetuned_path = None
finetuned_path = "finetuned/checkpoint.ckpt"

# 2D list to store realtime training data
sensors = [[] for i in range(num_sensors)]

# Bluetooth device for Myo Armband
selected_device = []

# Load MEAN and Standard Deviation for Standarization from Ninapro DB5 sEMG signals.
with open(config.std_mean_path, 'r') as f:
    params = json.load(f)

    
class Connection:
    
    client: BleakClient = None
    
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        EMG0: str,
        EMG1: str,
        EMG2: str,
        EMG3: str,
        CONTROL: str,
    ):
        self.loop = loop
        self.EMG0 = EMG0 # MyoCharacteristic0
        self.EMG1 = EMG1 # MyoCharacteristic1
        self.EMG2 = EMG2 # MyoCharacteristic2
        self.EMG3 = EMG3 # MyoCharacteristic3
        self.CONTROL = CONTROL
        self.connected = False
        self.connected_device = None
        self.model = get_finetune(config.save_path, config.prev_params, lr=0.0002, num_classes=len(GESTURES))
        self.current_sample = [[] for i in range(num_sensors)]
        self.last_data_time = time.time()
        self.count = 0

    """
        Handler for when BLE device is disconnected
    
    """
    def on_disconnect(self, client: BleakClient):
        self.connected = False
        print(f"Disconnected from {self.connected_device.name}!")

    """
        Callback right after BLE device is deisconnected
    
    """
    async def cleanup(self):
        print('Cleanup')
        # Terminates all communication attempts with BLE device
        if self.client:
            await self.client.stop_notify(EMG0)
            await self.client.stop_notify(EMG1)
            await self.client.stop_notify(EMG2)
            await self.client.stop_notify(EMG3)
            await self.client.disconnect()

    """
        Searches for nearby BLE devices or initiates connection with BLE device if chosen
    """
    
    async def manager(self):
        """Главный цикл управления состоянием подключения"""
        print("Starting connection manager.")
        while True:
            try:
                if not self.connected:
                    if await self.select_device():
                        await self.connect()
                    else:
                        await asyncio.sleep(2.0)
                # else:
                #     await asyncio.sleep(1.0)
                    
            except Exception as e:
                print(f"Manager error: {str(e)}")
                # self.connected = False
                # await self.cleanup()
                # await asyncio.sleep(1.0)

    async def connect(self):
        """Основной рабочий цикл после подключения"""
        try:
            if not await self.check_connection():
                return

            print("\n=== Начинаем сбор тренировочных данных ===")
            
            for gesture_id, gesture_name in enumerate(GESTURES):
                print(f"\n>>> Выполняйте жест: {gesture_name}...")
                
                await self.start_emg_stream()
                
                initial_len = len(sensors[0])
                while (len(sensors[0]) - initial_len) < SAMPLES_PER_GESTURE:
                    await asyncio.sleep(0.1)
                    
                await self.stop_emg_stream()
                
                print(f"Собрано {len(sensors[0]) - initial_len} семплов для {gesture_name}")

            print("\nОбработка данных...")
            inputs, outputs = realtime_preprocessing(
                sensors, 
                params_path=config.std_mean_path,
                num_classes=len(GESTURES), 
                window=window, 
                step=step_size
            )
            
            print("Обучение модели...")
            self.model.fit(
                inputs.reshape(-1, 8, window, 1).astype(np.float32),
                outputs,
                batch_size=realtime_batch_size,
                epochs=realtime_epochs
            )
            
            if finetuned_path:
                self.model.save_weights(finetuned_path)
            
            print("\n=== Начинаем распознавание жестов ===")
            await self.start_emg_stream(self.prediction_handler)
                
        except Exception as e:
            print(f"Ошибка в рабочем цикле: {str(e)}")


    async def check_connection(self):
        """Проверка активного соединения"""
        if not self.client or not self.client.is_connected:
            print("Нет активного соединения")
            return False
            
        #тест
        try:
            await self.client.write_gatt_char(
                self.CONTROL, 
                bytearray([1, 3, 2, 0, 0])
            )
            return True
        except Exception as e:
            print(f"Ошибка проверки соединения: {str(e)}")
            return False

    async def start_emg_stream(self, handler=None):
        """Активация потока EMG-данных"""
        if not handler:
            handler = self.training_handler
            
        for uuid in [self.EMG0, self.EMG1, self.EMG2, self.EMG3]:
            await self.client.start_notify(uuid, handler)

    async def stop_emg_stream(self):
        """Остановка потока EMG-данных"""
        print(f"Остановка считывания данных")

        for uuid in [self.EMG0, self.EMG1, self.EMG2, self.EMG3]:
            try:
                await self.client.stop_notify(uuid)
            except:
                pass
    """
        Selects and connects to a BLE device
    
    """
    async def select_device(self):
        print("Bluetooth LE hardware warming up...")
        await asyncio.sleep(2.0)

        print("Searching for Myo Armband...")
        try:
            devices = await discover()
            myo_device = next((d for d in devices if d.name and "Myo" in d.name), None)

            if not myo_device:
                print("Myo Armband not found.")
                return False

            print(f"Found Myo: {myo_device.name}, connecting...")
            self.client = BleakClient(myo_device.address)
            self.connected_device = myo_device

            await self.client.connect()
            print("Connected!")

            # 1. Активируем EMG-поток
            command = bytearray([1, 3, 2, 0, 0])
            await self.client.write_gatt_char(self.CONTROL, command)

            # 2. Тестовый прием данных
            received_data = False
            
            def test_handler(sender, data):
                nonlocal received_data
                print(f"Test data received: {data.hex()}")
                received_data = True

            for uuid in [self.EMG0, self.EMG1, self.EMG2, self.EMG3]:
                await self.client.start_notify(uuid, test_handler)

            for _ in range(30):  
                if received_data:
                    break
                await asyncio.sleep(0.1)

            for uuid in [self.EMG0, self.EMG1, self.EMG2, self.EMG3]:
                print("STOP notify")

                await self.client.stop_notify(uuid)

            if not received_data:
                print("No EMG data received")
                return False

            self.connected = True
            return True

        except Exception as e:
            print(f"Connection failed: {e}")
            await self.cleanup()
            return False

    

    # Handler for collecting 2 Sequential Sequence from Myo Armaband (MyoCharacteristics0)
    # Each Sequential Sequence contains 1 sample from each 8 sensors/channels
    def training_handler(self, sender: str, data: Any):
        # print(f"Training EMG Data from {sender}: {data.hex()}")  

        sequence_1, sequence_2 = getFeatures(data, twos_complement=True)
        for channel_idx in range(8):
            sensors[channel_idx].append(sequence_1[channel_idx])
            sensors[channel_idx].append(sequence_2[channel_idx])
  
  
    # Handler for collecting 2 Sequential Sequence from Myo Armaband
    # Each Sequential Sequence contains 1 sample from each 8 sensors/channels
    async def prediction_handler(self, sender: str, data: Any):
        try:
            sequence_1, sequence_2 = getFeatures(data, twos_complement=True)
            # print(f"Получены данные: {len(sequence_1)+len(sequence_2)} значений")  
            
            for channel_idx in range(8):
                self.current_sample[channel_idx].append(sequence_1[channel_idx])
                self.current_sample[channel_idx].append(sequence_2[channel_idx])
                
        except Exception as e:
            print(f"Ошибка в обработчике: {e}")
        
        if len(self.current_sample[0]) >= window:
            # Truncate self.current_samples to window size
            sEMG = np.array([samples[-window:] for samples in self.current_sample])
            
            # Apply Standarization to sEMG data
            for channel_idx in range(len(sEMG)):
                mean = params[str(channel_idx)][0]
                std = params[str(channel_idx)][1]
                sEMG[channel_idx] = (sEMG[channel_idx] - mean) / std
            
            # Optional cast input to float 32 (demand of microcontroller)
            sEMG = sEMG.astype(np.float32)
            
            # Get prediction results
            pred = realtime_pred(
                self.model,
                sEMG,
                num_channels=num_sensors,
                window_length=window
            )
            
            
            # Update prediction results
            print(GESTURES[pred])
            
            # Remove first 8 instance from self.current_samples to collect new data. (overlaps)
            self.current_sample = [samples[-(window-step_size):] for samples in self.current_sample]
            

def getFeatures(data, twos_complement=True):
    sequence_1 = []
    sequence_2 = []
    for i in range(8):
        if twos_complement==True and data[i] > 127:
            sequence_1.append(data[i]-256)
        else:
            sequence_1.append(data[i])
            
    for i in range(8,16):
        if twos_complement==True and data[i] > 127:
            sequence_2.append(data[i]-256)
        else:
            sequence_2.append(data[i])

    return sequence_1, sequence_2

#############
# App Main
#############
if __name__ == "__main__":

    # Create the event loop.
    loop = asyncio.get_event_loop()
    connection = Connection(loop, EMG0, EMG1, EMG2, EMG3, CONTROL) # EMG3
    try:
        asyncio.ensure_future(connection.manager())
        loop.run_forever()
    except KeyboardInterrupt:
        print("User stopped program.")
    finally:
        print("Disconnecting...")
        loop.run_until_complete(connection.cleanup())
