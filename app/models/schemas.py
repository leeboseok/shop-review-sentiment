from pydantic import BaseModel

class ReviewRequest(BaseModel):
    review_text: str

class FeedbackRequest(BaseModel):
    review_text: str
    sentiment: str
    feedback: str
