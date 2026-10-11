# Pocket-Lab
Machine Vision Course Project

## Project structure

```
Pocket-Lab/
├── app.py              # Flask app: open image, count and show cells
├── yolo_detection.py   # Finds cells with the trained YOLO model
├── postprocessing.py   # Corrects YOLO's red cell boxes with a cell mask
├── cell_detection.py   # Older OpenCV-only detection (no longer used by the app)
├── requirements.txt
├── models/
│   └── blood_cells.pt  # Trained YOLO model
├── training/           # Scripts to build the dataset, train and evaluate
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

Then open http://127.0.0.1:5001

## Next steps

- Explore using pre-trained models for cell detection (models already trained on blood cell images), and compare them with our own YOLO model.
