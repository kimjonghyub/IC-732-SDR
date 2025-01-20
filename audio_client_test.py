import asyncio
import logging
import struct
import numpy as np
import sounddevice as sd
from opuslib import Decoder

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

AUDIO_SAMPLE_RATE = 8000  
AUDIO_CHANNELS = 1 
FRAME_DURATION = 10 # ms       
FRAME_SIZE = int(AUDIO_SAMPLE_RATE * FRAME_DURATION / 1000)
decoder = Decoder(AUDIO_SAMPLE_RATE, AUDIO_CHANNELS)


QUEUE_MAXSIZE = 1

async def process_audio_queue(audio_queue):
   
    with sd.OutputStream(samplerate=AUDIO_SAMPLE_RATE, channels=AUDIO_CHANNELS, dtype='int16') as stream:
        while True:
            audio_samples = await audio_queue.get()
            stream.write(audio_samples)
            #logging.debug(f"Played audio frame of size: {len(audio_samples)} samples")
            audio_queue.task_done()

async def connect_to_audio_server(host, port):
    writer = None
    audio_queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)

    
    asyncio.create_task(process_audio_queue(audio_queue))

    try:
        logging.info(f"Attempting to connect to audio server at {host}:{port}")
        reader, writer = await asyncio.open_connection(host, port)
        logging.info(f"Connected to audio server at {host}:{port}")

        while True:
            try:
                
                frame_size_data = await reader.readexactly(4)
                frame_size = struct.unpack(">I", frame_size_data)[0]

                
                encoded_audio = await reader.readexactly(frame_size)

                
                decoded_audio = decoder.decode(encoded_audio, FRAME_SIZE)
                audio_samples = np.frombuffer(decoded_audio, dtype=np.int16)

               
                if audio_queue.full():
                    audio_queue.get_nowait()
                audio_queue.put_nowait(audio_samples)

                logging.debug(f"Received and queued audio frame of size: {len(audio_samples)} samples")
            except asyncio.IncompleteReadError:
                logging.error("Audio server disconnected unexpectedly.")
                break
            except Exception as e:
                logging.error(f"Error while receiving or decoding audio: {e}")
                break
    except Exception as e:
        logging.error(f"Error connecting to audio server: {e}")
    finally:
        if writer:
            writer.close()
            await writer.wait_closed()
        logging.info("Audio connection closed.")

if __name__ == "__main__":
    asyncio.run(connect_to_audio_server("222.117.38.98", 5633))
