# Pocket-Lab
Machine Vision Course Project

## Project structure

```
Pocket-Lab/
├── app.py              # Flask app + OpenCV routes
├── requirements.txt
├── templates/
│   └── index.html
└── static/
    ├── app.js
    ├── style.css
    └── uploads/        # created at runtime, not tracked by git
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000
