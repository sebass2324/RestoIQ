from app import create_app
from models import db
from models.venta import Venta

app = create_app()
with app.app_context():
    TU_USER_ID = 6
    v_manana = Venta.query.filter_by(user_id=TU_USER_ID).filter(Venta.lluvia_manana_mm.isnot(None)).count()
    v_noche  = Venta.query.filter_by(user_id=TU_USER_ID).filter(Venta.lluvia_nocturna_mm.isnot(None)).count()
    print(f"Con lluvia_manana_mm: {v_manana}  |  Con lluvia_nocturna_mm: {v_noche}")