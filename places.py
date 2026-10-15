from enum import Enum

class Location(str, Enum):
    Home = "Home"
    Work = "Work"
    Restaurant = "Restaurant"
    School = "School"
    Recreation = "Recreation"
    Errand = "Errand"
    Hospital = "Hospital"