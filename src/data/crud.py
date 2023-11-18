from . import models
from sqlalchemy.orm import Session


def create_qr_history(db: Session, qr_history: models.QrHistory):
    db.add(qr_history)
    db.commit()
    return qr_history


def get_qr_histories(db: Session, offset: int = 0):
    return db.query(models.QrHistory).offset(offset).all()
