from flask import Blueprint, render_template, request, redirect, session
from werkzeug.security import generate_password_hash, check_password_hash
from models.models import User, db

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/account', methods=['GET', 'POST'])
def account():
    """Page de gestion des comptes utilisateurs"""
    message = ""
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'edit' and user:
            # Modification du profil utilisateur
            new_username = request.form.get('edit_username', '').strip()
            new_email = request.form.get('edit_email', '').strip()
            new_password = request.form.get('edit_password', '').strip()
            if new_email and new_email != user.email and User.query.filter_by(email=new_email).first():
                message = "Cet email est déjà utilisé."
            else:
                if new_username:
                    user.username = new_username
                if new_email:
                    user.email = new_email
                if new_password:
                    user.password = generate_password_hash(new_password)
                
                db.session.commit()
                message = "Profil mis à jour."
        else:
            # Inscription ou connexion
            username = request.form.get('username', '').strip()
            email = request.form.get('email', '').strip()
            password = request.form.get('password', '').strip()
            if action == "register":
                # Inscription
                if User.query.filter_by(email=email).first():
                    message = "Email déjà pris."
                else:
                    if not username:
                        # Génération d'un nom d'utilisateur automatique
                        temp_user = User(username="temp", email=email, password=generate_password_hash(password))
                        db.session.add(temp_user)
                        db.session.flush()
                        temp_user.username = f"user{temp_user.id}"
                        db.session.commit()
                        message = "Inscription réussie. Connectez-vous."
                        user = None
                        return render_template('account.html', message=message, user=None)
                    else:
                        hashed_pw = generate_password_hash(password)
                        user = User(username=username, email=email, password=hashed_pw)
                        db.session.add(user)
                        db.session.commit()
                        message = "Inscription réussie. Connectez-vous."
            elif action == "login":
                # Connexion
                user = User.query.filter_by(email=email).first()
                if user and check_password_hash(user.password, password):
                    session['user_id'] = user.id
                    return redirect('/account')
                else:
                    message = "Identifiants incorrects."
            user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    return render_template('account.html', message=message, user=user)

@auth_bp.route('/logout')
def logout():
    """Déconnexion de l'utilisateur"""
    session.pop('user_id', None)
    return redirect('/account')