## Dataset
- Unzip mnist.zip to `./mnist`
    ```sh
    unzip mnist.zip -d ./mnist
    ```
- Folder structure
    ```
    .
    ├── mnist
    ├── diffusion_process.py
    ├── model.py
    ├── Readme.md
    ├── requirements.txt
    ├── score.py
    └── train.py
    ```

## Environment
- Python 3.6.13 or later version
    ```sh
    pip install -r requirements.txt
    ```

## Train
```sh
python train.py
```
The model state save in `save_model` folder.

## Get Fid Score
```sh
python score.py
```

## Get Diffusion Process
```sh
python diffusion_process.py
```
You can get a diffusion process png.
