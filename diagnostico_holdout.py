from app import create_app
from models import db
from models.venta import Venta

app = create_app()
with app.app_context():
    TU_USER_ID = 5  # ajusta al tuyo real, confirmado
    borradas = Venta.query.filter_by(user_id=TU_USER_ID).delete()
    db.session.commit()
    print(f"Se borraron {borradas} filas.")