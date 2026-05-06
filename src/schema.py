from pydantic import BaseModel, Field
from typing import List

class LegalMetadata(BaseModel):
    parties: List[str] = Field(
        default=[], 
        description="Names of people, companies, or entities involved in the text."
    )
    clauses: List[str] = Field(
        default=[], 
        description="Titles or types of legal clauses identified (e.g., 'Governing Law', 'Indemnification')."
    )
    obligations: List[str] = Field(
        default=[], 
        description="Specific duties, payments, or actions required by the parties."
    )