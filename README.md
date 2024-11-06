# DIY IC732 Panadapter for Rtl-sdr!!!

## How to Connect IF (Intermediate Frequency) Signals to RTL-SDR
![screen1](https://github.com/user-attachments/assets/317b3666-b436-4205-a144-0ab7d4aebcaf)  
### 1. Audio AF input image  

![screen2-sdr](https://github.com/user-attachments/assets/810e9948-41f5-4798-9ed4-9b8edb0fad62)  
### 2. Rtl-sdr input image  

![if text](https://github.com/user-attachments/assets/a15fb88c-7c2a-4a7d-9fce-3344c845bfea)  
### 3. IF Frequency 69.011500mHz  

![main pcb sch](https://github.com/user-attachments/assets/43137579-a2c5-452a-b954-d373807368e9)  
### 4. IF Point  

![main pcb](https://github.com/user-attachments/assets/df089307-b83f-4586-b883-81e8b7419d56)  
### 5. IF Point image  

![main pcb_1](https://github.com/user-attachments/assets/8fdc47e6-bedb-445c-acec-49525ca8002b)  
### 6. IF Point image  

![main pcb img](https://github.com/user-attachments/assets/b0b3e42b-2834-4e4f-8673-aa616a83c11a)  
### 7. IF Point image  

![main pcb img_1](https://github.com/user-attachments/assets/c69d15d5-8721-476a-b461-44152a722c75)  
### 8. IF Point image  ==> Connect a 5pF capacitor in series between the IF point and the RF cable.  

![main pcb img_if](https://github.com/user-attachments/assets/0817e54d-c5c6-42dc-839f-67c2420e6a92)  
### 9. IF Point image  

![smc1](https://github.com/user-attachments/assets/c635ed1e-5573-49bf-b10f-caebb3e44fa3)  
### 10. RF Connector image  

![smc2](https://github.com/user-attachments/assets/76122360-781d-4bdf-b523-391ba51e0ba9)  
### 11. RF Connector image  

![smc3](https://github.com/user-attachments/assets/c3f558a5-a161-4832-8dfc-4b8593a7b1d3)  
### 12. Wire Connection  

![IMG_2948](https://github.com/user-attachments/assets/a8e56acf-a71c-4a2c-83f9-058eea14f5fe)  
### 13. HDSDR   


# Original Author Program Source  
Tiny Python Panadapter (QST, April, 2014)
Martin Ewing AA6E
Revised May 20, 2014

# Pre-installation for Raspberry Pi 4 Bookworm Version  
pi@raspberrypi:~ $sudo apt-get install python3-pip

pi@raspberrypi:~ $pip install pygame --break-system-packages

pi@raspberrypi:~ $pip install numpy --break-system-packages

pi@raspberrypi:~ $sudo pip install pyrtlsdr --break-system-packages

pi@raspberrypi:~ $sudo apt-get install portaudio19-dev -y

pi@raspberrypi:~ $sudo apt-get install rtl-sdr librtlsdr-dev

pi@raspberrypi:~ $pip install pyaudio --break-system-packages

pi@raspberrypi:~ $sudo apt-get install git -y

pi@raspberrypi:~ $sudo apt-get install fontconfig

pi@raspberrypi:~ $wget https://github.com/Hamlib/Hamlib/releases/download/4.5.5/hamlib-4.5.5.tar.gz

pi@raspberrypi:~ $tar -xvzf hamlib-4.5.5.tar.gz

pi@raspberrypi:~ $sudo apt-get install build-essential automake autoconf swig 

pi@raspberrypi:~ $cd hamlib-4.5.5

pi@raspberrypi:~ /hamlib-4.5.5 $sudo  ./configure --with-python-binding PYTHON=$(which python3) --prefix=/usr

pi@raspberrypi:~ /hamlib-4.5.5 $sudo make

pi@raspberrypi:~ /hamlib-4.5.5 $sudo make install 

### Source Code Sections to Modify and Compile in Hamlib-4.5.5  
File Path ==> /home/pi/hamlib-4.5.5/src/rig.c  
Content to Modify ==> rs->cache.timeout_ms = 50; 

File Path ==> /home/pi/hamlib-4.5.5/rigs/icom/ic737.c  
Content to Modify ==> .serial_rate_max =  9600,  

### Recompile and Install  
pi@raspberrypi:~/hamlib-4.5.5 $sudo make install  

## Run the Program
pi@raspberrypi:~ $sudo git clone http://https://github.com/kimjonghyub/IC732-SDR.git  
pi@raspberrypi:~ /cd IC732-SDR  
pi@raspberrypi:~ /IC732-SDR $python iq.py  

## Keyboard Setup  
Pressed 1(!). STEP  
Pressed 2(@). BAND  
Pressed 3(#). MODE  
Pressed 4($). VFO A/B  
Pressed Left. Freq down  
Pressed Right. Freq Up  
Mouse Wheel Scrol. Down. Freq Down  
Mouse Wheel Scroll Up. Freq Up  

## SETUP(iq_opt.py)  
![iqopt](https://github.com/user-attachments/assets/bc8b9e26-a344-44b1-b694-dab45d25ad15)  
hamlib_device  
hamlib_rigtype  
index  
rtl_frequency  
source_rtl  



## Reference Website Address  
https://buildthings.wordpress.com/ham-radio-raspberry-pi-aa6e-tiny-python-panadapter/
