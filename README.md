# Offline Web Application Deployment Guide

You can deploy our web application offline on your local computer by following these steps:

## 1. Environment Setup

First, create and set up the Python environment:

```bash
conda create --name myenv python=3.10
conda activate myenv
pip install -r requirements.txt
```

## 2. Launch the Application

Run the application with:

```bash
python app.py
```

## 3. Using the Web Application

1. Open your web browser and navigate to the local address shown in the terminal (typically `[http://127.0.0.1:5000](http://localhost:5050/)`)
2. Enter your:
   - API Key
   - Dataset
   - Query
3. Submit to receive your answers!
