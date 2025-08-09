📩 Spam SMS Detection (Flask App)
A simple yet effective Machine Learning web application built using Flask that classifies SMS messages as Spam or Not Spam.
The model is trained using Naive Bayes and text features extracted via CountVectorizer.

🚀 Features
User-friendly web interface to enter SMS text.

Real-time prediction of whether the message is spam or not.

Efficient model loading — trained once and reused.

Built with Flask for backend and HTML/CSS for frontend.

Uses Multinomial Naive Bayes for classification.

📂 Project Structure
Spam-SMS-Detection/
│
├── app.py                 
├── train_model.py          
├── NB_spam_model.pkl      
├── count_vectorizer.pkl    
├── spam.csv               
│
├── templates/
│   ├── home.html           
│   ├── result.html        
│
└── static/
    └── style.css           


📌 Future Improvements
Use TF-IDF Vectorizer for better accuracy.

Deploy on Heroku/Render for online access.

Add SMS length, punctuation count, and word diversity as features.

Use deep learning models like LSTM for better contextual understanding.

