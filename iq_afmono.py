#!/usr/bin/env python

import sys, time, threading
import queue
import pyaudio as pa

# Global variables
led_underrun_ct = 0
cbcount = 0
MAXQUEUELEN = 128
cbqueue = queue.Queue(MAXQUEUELEN)
cbskip_ct = 0
queueLock = threading.Lock()
cbfirst = 1

audio = pa.PyAudio() #af mono

def pa_callback_iqin(in_data, f_c, time_info, status):
    global cbcount, cbqueue, cbskip, cbskip_ct
    global led_underrun_ct, queueLock, cbfirst
    
    cbcount += 1
    #if cbcount % 10 == 0:
        #print(f"Callback count: {cbcount}, Data length: {len(in_data)}")

    if status == pa.paInputOverflow:
        led_underrun_ct = 1  # signal LED "underrun" (actually overflow)

    # Check if we need to skip the buffer based on `cbskip`
    if cbskip > 0:  
        if cbskip_ct >= cbskip:
            cbskip_ct = 0
            return (None, pa.paContinue)  # Discard this buffer
        else:
            cbskip_ct += 1  
    elif cbskip < 0:
        if cbskip_ct >= -cbskip:
            cbskip_ct = 0  
        else:
            cbskip_ct += 1
            return (None, pa.paContinue)  # Discard this buffer

    if cbfirst > 0:
        cbfirst -= 1
        return (None, pa.paContinue)  # Discard initial buffers

    try:
        queueLock.acquire()
        if cbqueue.full():
            cbqueue.get_nowait()
        cbqueue.put_nowait(in_data)
        queueLock.release()
    except queue.Full:
        print("ERROR: Internal queue is full. Reconfigure to use less CPU.")
        sys.exit()
    return (None, pa.paContinue)

class DataInput(object):
    """ Set up audio input with callbacks. """
    def __init__(self, opt=None):
        self.audio = pa.PyAudio()
        print()
        self.Restart(opt)
        
    def Restart(self, opt):  
        global cbqueue, cbskip

        cbskip = opt.skip
        print()
        
        # Set up mono (1 channel) / 48KHz IQ input channel
        if opt.index < 0:
            defdevinfo = self.audio.get_default_input_device_info()
            print("Default device index is %d; id='%s'" % 
                  (defdevinfo['index'], defdevinfo['name']))
            af_using_index = defdevinfo['index']
        else:
            af_using_index = opt.index
            devinfo = self.audio.get_device_info_by_index(af_using_index)
            print("Using device index %d; id='%s'" % 
                  (devinfo['index'], devinfo['name']))

        try:
            # Verify this mode is supported
            support = self.audio.is_format_supported(
                input_format=pa.paInt16,       
                input_channels=1,              # 1 channel for mono
                rate=opt.sample_rate,  
                input_device=af_using_index)
        except ValueError as e:
            print("ERROR: audio format not supported:", e)
            sys.exit()

        print("Requested audio mode is supported:", support)
        self.afiqstream = self.audio.open(
            format=pa.paInt16,          
            channels=1,                 # 1 channel for mono
            rate=opt.sample_rate,       
            #frames_per_buffer=opt.buffers * opt.size,
            frames_per_buffer=opt.buffers * opt.size,
            input_device_index=af_using_index,
            input=True,
            stream_callback=pa_callback_iqin)
        
    def get_queued_data(self):
        timeout = 40
        while cbqueue.qsize() < 4:
            timeout -= 1
            if timeout <= 0: 
                print("Timeout waiting for queue to become non-empty!")
                sys.exit()
            time.sleep(.1)
        queueLock.acquire()
        data = cbqueue.get(True, 4.)
        queueLock.release()
        return data
    
    def CPU_load(self):
        return self.afiqstream.get_cpu_load()

    def isActive(self):
        return self.afiqstream.is_active()
        
    def Start(self):  
        self.afiqstream.start_stream()

    def Stop(self):  
        self.afiqstream.stop_stream()
    
    def CloseStream(self):
        self.afiqstream.stop_stream()
        self.afiqstream.close()

    def Terminate(self):  
        self.audio.terminate()

class DataOutput(object):
    def __init__(self, sample_rate, chunk_size):
        self.output_stream = audio.open(
            format=pa.paInt16,
            channels=1,
            rate=sample_rate,
            output=True,
            frames_per_buffer=chunk_size
        )

    def play(self, data):
        self.output_stream.write(data)

    def stop(self):
        self.output_stream.stop_stream()
        self.output_stream.close()

if __name__ == '__main__':
    print('debug')  
