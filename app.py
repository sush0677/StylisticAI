import warnings
warnings.filterwarnings("ignore", message=".*beta.*renamed internally to.*")
warnings.filterwarnings("ignore", message=".*gamma.*renamed internally to.*")
warnings.filterwarnings("ignore")

from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import torch
import os
from functools import wraps
from flask_dance.contrib.google import make_google_blueprint, google
from flask_dance.contrib.github import make_github_blueprint, github
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
db = SQLAlchemy(app)

# OAuth setup for Google
google_bp = make_google_blueprint(
    client_id="995635811403-1k0eom8s4f8q48l1qe7ped2nl4sm0gb9.apps.googleusercontent.com",
    client_secret="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",
    redirect_to='google_login',
    scope=["profile", "email"]
)
app.register_blueprint(google_bp, url_prefix="/login")

# OAuth setup for GitHub
github_bp = make_github_blueprint(
    client_id="Ov23li9u80YXGR6bE49m",
    client_secret="f913752e46957d467a7af22162006e994cdf0559",
    redirect_to='github_login'
)
app.register_blueprint(github_bp, url_prefix="/login")

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)

with app.app_context():
    db.create_all()

# Define the Transformer-based model
class DressRecommendationModel(nn.Module):
    def __init__(self, num_classes):
        super(DressRecommendationModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs[1]  # CLS token output
        x = self.dropout(cls_output)
        x = self.fc(x)
        return x

def train_model():
    # Load the preprocessed data
    X_train, X_val, y_train, y_val, label_encoder = torch.load('preprocessed_data.pth')

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Initialize the model
    num_classes = len(label_encoder.classes_)
    model = DressRecommendationModel(num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    # Training loop
    epochs = 3
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, labels = batch
                outputs = model(input_ids)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

    print("Model training completed.")
    return model, label_encoder

# Train the model on app start
model, label_encoder = train_model()

def generate_recommendation(input_text):
    """Generate a fashion recommendation based on the input text using the transformer model."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Tokenize and prepare input
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=50)
    attention_mask = torch.ones_like(input_ids)

    # Predict using the model
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)
    
    # Get the predicted class and decode to recommendation text
    predicted_class = torch.argmax(output, dim=1).item()
    recommendation = label_encoder.inverse_transform([predicted_class])[0]

    return recommendation

def login_required(f):
    """Decorator to require login for specific routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    """Render the home page."""
    username = session.get('username')
    return render_template('index.html', username=username)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_password)
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except:
            flash('Username already exists.', 'danger')
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials, please try again.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))

@app.route('/google_login')
def google_login():
    if not google.authorized:
        return redirect(url_for("google.login"))
    resp = google.get("/oauth2/v2/userinfo")
    assert resp.ok, resp.text
    info = resp.json()
    username = info["name"]
    session['username'] = username
    flash('Logged in with Google successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/github_login')
def github_login():
    if not github.authorized:
        return redirect(url_for("github.login"))
    resp = github.get("/user")
    assert resp.ok, resp.text
    info = resp.json()
    username = info["login"]
    session['username'] = username
    flash('Logged in with GitHub successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/recommendation', methods=['GET', 'POST'])
@login_required
def recommendation():
    """Render the fashion recommendation page."""
    generated_trend = None
    if request.method == 'POST':
        user_input = request.form['user_input']
        generated_trend = generate_recommendation(user_input)
    username = session.get('username')
    return render_template('recommendation.html', generated_trend=generated_trend, username=username)

@app.route('/questions', methods=['GET', 'POST'])
@login_required
def questions():
    """Render the fashion questionnaire page and store answers."""
    recommendation = None
    if request.method == 'POST':
        step = request.form.get('step')
        if step == "1":
            session['gender'] = request.form.get('gender')
        elif step == "2":
            session['personal_style'] = request.form.get('personal_style')
        elif step == "3":
            session['color_preference'] = request.form.get('color_preference')
        elif step == "4":
            session['wardrobe_needs'] = request.form.get('wardrobe_needs')
        elif step == "5":
            session['lifestyle'] = request.form.get('lifestyle')

            # Generate a recommendation based on all stored answers
            prompt = f"You are a {session['gender']} with a {session['personal_style']} style who prefers {session['color_preference']} colors. You need outfits for {session['wardrobe_needs']} and your typical day involves {session['lifestyle']}."
            recommendation = generate_recommendation(prompt)

    return render_template('questions.html', recommendation=recommendation)

@app.route('/about')
def about():
    """Render the about page."""
    username = session.get('username')
    return render_template('about.html', username=username)

@app.route('/contact')
def contact():
    """Render the contact page."""
    username = session.get('username')
    return render_template('contact.html', username=username)

if __name__ == '__main__':
    app.run(debug=True)
