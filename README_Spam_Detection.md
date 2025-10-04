# 📧 Spam Detection Using Naive Bayes

A machine learning project that implements a Naive Bayes classifier to detect spam messages. This project demonstrates the power of probabilistic classification for text-based spam detection.

## 📋 Overview

This project uses the Naive Bayes algorithm to classify messages as spam or ham (not spam). The classifier learns from a dataset of labeled messages and can predict whether new messages are spam with high accuracy.

## ✨ Features

- **Naive Bayes Classification**: Implements probabilistic text classification
- **Text Processing**: Handles message preprocessing and feature extraction
- **Model Evaluation**: Includes accuracy metrics and performance analysis
- **Jupyter Notebook**: Interactive analysis and visualization
- **Real Dataset**: Uses authentic spam/ham message dataset

## 🎯 What is Naive Bayes?

Naive Bayes is a probabilistic machine learning algorithm based on Bayes' Theorem. It's particularly effective for:
- Text classification
- Spam filtering
- Sentiment analysis
- Document categorization

The "naive" assumption is that all features are independent, which simplifies computation while maintaining good performance.

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Required libraries (pandas, numpy, scikit-learn, matplotlib)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/alicenjr/Spam-Detection-Using-Naive-Bayes.git
cd Spam-Detection-Using-Naive-Bayes
```

2. Install required packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

3. Open the Jupyter notebook:
```bash
jupyter notebook spam_detection.ipynb
```

## 📊 Dataset

The project uses `spam.csv` containing:
- Message text
- Label (spam/ham)
- Multiple features for classification

## 🔧 Implementation

The implementation includes:

1. **Data Loading & Exploration**
   - Load spam.csv dataset
   - Exploratory data analysis
   - Data visualization

2. **Data Preprocessing**
   - Text cleaning and normalization
   - Tokenization
   - Feature extraction (TF-IDF or CountVectorizer)

3. **Model Training**
   - Split data into training and testing sets
   - Train Naive Bayes classifier
   - Parameter tuning

4. **Evaluation**
   - Accuracy score
   - Confusion matrix
   - Precision, recall, and F1-score
   - ROC curve analysis

## 📈 Results

The Naive Bayes classifier typically achieves:
- High accuracy (>95% on test data)
- Fast training and prediction times
- Good generalization to new messages

## 💡 Key Concepts

**Bayes' Theorem:**
```
P(spam|message) = P(message|spam) × P(spam) / P(message)
```

The classifier calculates probabilities for spam and ham, then selects the class with higher probability.

## 🔍 Use Cases

- Email spam filtering
- SMS spam detection
- Comment moderation
- Fraud detection
- Content filtering

## 📁 Project Structure

```
├── spam_detection.ipynb    # Main Jupyter notebook with implementation
├── spam.csv               # Dataset with labeled messages
└── README.md              # Project documentation
```

## 🛠️ Technologies Used

- **Python**: Programming language
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning library
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter**: Interactive development environment

## 🎓 Learning Outcomes

This project demonstrates:
- Text preprocessing techniques
- Naive Bayes algorithm implementation
- Model evaluation metrics
- Classification pipeline development
- Feature engineering for text data

## 🤝 Contributing

Contributions are welcome! You can:
- Improve the preprocessing pipeline
- Try different Naive Bayes variants (Multinomial, Bernoulli, Gaussian)
- Add more evaluation metrics
- Experiment with feature extraction methods

## 📝 License

This project is open-source and available for educational and commercial use.

## 👨‍💻 Author

**alicenjr** - [GitHub Profile](https://github.com/alicenjr)

---

⭐ Star this repo if you find it useful!
