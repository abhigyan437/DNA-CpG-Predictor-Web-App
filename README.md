# DNA-CpG-Predictor-Web-App

This repository contains a Streamlit-based web application that predicts the number of "CG" motifs (CpG sites) in a given DNA sequence using pre-trained PyTorch models.

📁 Project Files

| File Name                  | Description                                                    |
| -------------------------- | -------------------------------------------------------------- |
| `app.py`                   | Main Streamlit app script (UI + model inference logic)         |
| `model.pth`                | Model trained on **fixed-length DNA sequences** (length = 128) |
| `model_unequal_length.pth` | Model trained on **variable-length DNA sequences**             |



⚙️ Backend Logic (Streamlit App)
The app accepts a DNA sequence input from the user (consisting of characters A, C, G, T, and N).

1) It checks the length of the sequence:
2) If the sequence length is exactly 128, it uses the model.pth (fixed-length model).
3) Otherwise, it uses model_unequal_length.pth (variable-length model).
4) The model output (predicted number of CpG sites) is displayed on the page.


🚀 Running the App
Open your terminal in the project directory and run:

streamlit run app.py

This will launch the web app in your browser.


👩‍🔬 User Guide (Front-End Flow)
1) Enter a DNA sequence into the input box on the Streamlit web page.
2) The app automatically selects the correct model based on sequence length.
3) The predicted CpG (CG) count is shown immediately below the input, on clicking Enter.


✅ Notes
1) Input validation is enforced (only valid DNA bases allowed).
2) The models are trained using PyTorch.
3) This project supports both equal-length and variable-length sequence predictions.
