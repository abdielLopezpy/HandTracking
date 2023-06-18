# Hand Tracking

Este proyecto utiliza la biblioteca Mediapipe y OpenCV para realizar un seguimiento de la mano en tiempo real y controlar el mouse. El código está escrito en Python.

## Demo
![Ejemplo de imagen](imagen.png)

## Requisitos

- Python 3.x
- OpenCV
- Mediapipe
- PyAutoGUI

## Instalación

1. Clona este repositorio o descarga los archivos.

2. Abre una terminal y navega hasta el directorio del proyecto.

3. Crea un entorno virtual (opcional pero recomendado) en macOS y Linux o Windows:

    ```bash
    python3 -m venv venv
    ```
    ```bash
    py -m venv venv
    ```
4. Activa el entorno virtual:

    ```bash
    source venv/bin/activate
    ```
    ```bash
    venv\Scripts\activate
    ```
5. Instala los paquetes necesarios:

    ```bash
    pip3 install -r requirements.txt
    ```
    ```bash
    pip install -r requirements.txt
    ```
   
6. Ejecuta el script:

    ```bash
    python3 main.py
    ```
    ```bash
    py main.py
    ```
   
## Uso

- Para mover el mouse, mueve tu mano en el aire.
- Para hacer clic derecho, cierra la mano.


## Licencia

[MIT](https://choosealicense.com/licenses/mit/)

## Creado por Alejandro López