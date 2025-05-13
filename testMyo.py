import asyncio
from bleak import BleakClient

CONTROL_UUID = "d5060401-a904-deb9-4748-2c7f4a124842" 
EMG_UUIDS = [
    "d5060105-a904-deb9-4748-2c7f4a124842",  # EMG0
    "d5060205-a904-deb9-4748-2c7f4a124842",  # EMG1
    "d5060305-a904-deb9-4748-2c7f4a124842",  # EMG2
    "d5060405-a904-deb9-4748-2c7f4a124842",  # EMG3
]

async def connect_myo(address):
    def emg_handler(sender, data):
        """Обработчик сырых EMG-данных (8 сенсоров, 2 значения на пакет)."""
        print(f"EMG Data from {sender}: {data.hex()}")  # данные в hex

    try:
        async with BleakClient(address) as client:
            print(f"Connected to Myo: {client.is_connected}")

            command = bytearray([1, 3, 2, 0, 0])  # включение EMG
            await client.write_gatt_char(CONTROL_UUID, command)

            for uuid in EMG_UUIDS:
                await client.start_notify(uuid, emg_handler)

            print("Receiving EMG data...")
            await asyncio.sleep(300)

            for uuid in EMG_UUIDS:
                await client.stop_notify(uuid)

    except Exception as e:
        print(f"Error: {e}")

MYO_MAC = "FF:19:42:25:DB:F3"  
asyncio.run(connect_myo(MYO_MAC))