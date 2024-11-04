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
### 8. IF Point image  

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







Tiny Python Panadapter (QST, April, 2014)
Martin Ewing AA6E
Revised May 20, 2014

RELEASE NOTES

June 9, 2014
Version 0.3.6

This version contains support for radios using the Si570 DDS chip as VFO.  For example, in various
SoftRock radios, like the RxTx Ensemble.  A pure Python USB interface is provided that requires the
libusb-1.0 wrapper from https://pypi.python.org/pypi/libusb1/1.2.0 .

The current edition of the supplementary material for the QST article is now provided in the GIT repository
and in the downloadable zip file in ODT (Open Document Text) and PDF (Portable Document) formats.

May 20, 2014
Version 0.3.0

The enclosed file "TPP_qst_suppl_2014_04a.odt" (with .pdf and .doc versions) is an expanded version of the article printed in QST (April, 2014).  It has been updated, but it is probably not quite current, since the project has continued in development after publication.

To use this version, copy the tinypythonpanadapter-code directory to a convenient location on your computer.  Follow directions in the document above to load or build the required resources for your platform.

The most up-to-date information on the TPP project is to be found on-line:

1. Program files, git repository, and related technical material are permanently stored at SourceForge.net - https://sourceforge.net/projects/tinypythonpanadapter/ .

2. Project news, discussion of user issues is available on a SourceForge mailing list.  You can mail questions to tinypythonpanadaptor-discussion@lists.sourceforge.net . 

For a free subscription, sign up at https://lists.sourceforge.net/lists/listinfo/tinypythonpanadapter-discussion .

3. A "lab notebook" about the TPP project with lots of technical and installation information is provided at http://www.aa6e.net/wiki/Tiny_Python_Panadapter .  

4. Text files (e.g. Python source) in the enclosed zip archive may be in Linux format. They can be read with Microsoft Word or similar software or converted to Windows format using the gedit text editor (available for Linux or Windows at https://wiki.gnome.org/Apps/Gedit).


