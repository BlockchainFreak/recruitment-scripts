from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, JSON, Text

Base = declarative_base()

class StructuredResume(Base):
    __tablename__ = 'structured_resumes'

    id = Column(String(length=64), primary_key=True)
    content = Column(Text)
    structured = Column(JSON)

    def __repr__(self):
        return f"<StructuredResume(content={self.content}, structured={self.structured})>"

# # Create all tables in the Database. This is equivalent to "Create Table"
# from database import engine
# Base.metadata.create_all(engine)