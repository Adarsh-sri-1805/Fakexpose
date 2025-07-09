# 🕵️ Fakexposed – Real or Fake News Detector

Fakexposed is a Streamlit web app that uses machine learning to detect whether a news article or headline is real or fake.

## 🚀 Live Features
- Trained on [Kaggle Fake/Real News dataset]((https://www.kaggle.com/code/therealsampat/fake-news-detection/input))
- Uses TF-IDF vectorization + Logistic Regression
- Achieves 99% accuracy on known dataset
- Optional GPT-powered fact-checking for deeper analysis (experimental)
  
## Tools used
| Tool / Library              | Purpose                                          |
| --------------------------- | ------------------------------------------------ |
| **Python** 🐍               | Core programming language                        |
| **Pandas** 📊               | Data preprocessing and manipulation              |
| **Scikit-learn** 🤖         | TF-IDF vectorization & Logistic Regression model |
| **Streamlit** 🧼            | Building interactive web app                     |
| **Matplotlib / Seaborn** 🎨 | Visualization of confusion matrix and accuracy   |

## 🧠 How it Works
- Cleaned and labeled fake & real news
- TF-IDF vectorizes the combined text
- Logistic Regression classifier predicts real (0) or fake (1)

## ✅ Example Predictions

| Input from Dataset | Prediction | Confidence |
|---------------------|------------|------------|
| Real article from `True.csv` | 🟢 Real | 97.5% |
| Fake article from `Fake.csv` | 🔴 Fake | 96.3% |

## 💻 Run it Locally

```bash
pip install -r requirements.txt
streamlit run fakexposed_app.py
