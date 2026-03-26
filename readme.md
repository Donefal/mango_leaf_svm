## Cara setup
- Download dataset: https://www.kaggle.com/datasets/warcoder/mango-leaf-disease-dataset?select=MangoLeafBD+Dataset 
- Ambil data dan ubah nama folder
  - "Anthracnose" menjadi "anthracnose"
  - "Healthy" menjadi "healthy"
  - "Powdery Mildew" menjadi "powdery_mildew"
- Masukkan ketiga folder tersebut kesebuah folder bernama `datasets` dan simpan di root
- Pastikan struktur folder sesuai dengan file-tree dibawah
- Buat vitual environment python dengan command: `python -m venv .venv`
- Aktifkan virtual environment dengan command: `./.venv/Scripts/activate`
- Install dependencies dengan command: `pip install -r requirements.text`
  - Jika terdapat error coba ubah setting pada `./.venv/pyvenv.cfg` pada `include-system-site-packages` menjadi `true`
- Jalankan sesuka hati pada `main.ipynb`, jangan lupa untuk memilih kernel yang sesuai dengan venv nya

### Final File tree
``` tree
.
└── root/
    ├── .venv/
    ├── datasets/
    │   ├── anthracnose/
    │   ├── healthy/
    │   └── powdery_mildew/
    ├── for_predict/
    ├── models/
    ├── src/
    │   ├── __init__.py
    │   ├── feature_extraction.py
    │   ├── predict.py
    │   ├── preprocess.py
    │   └── train.py
    ├── .gitignore
    ├── main.ipynb
    ├── readme.md
    └── requirements.txt
```