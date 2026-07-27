from flask_login import LoginManager

login_manager = LoginManager()

@login_manager.user_loader
def load_user(user_id):
    from models.users import Usuario
    return Usuario.query.get(int(user_id))