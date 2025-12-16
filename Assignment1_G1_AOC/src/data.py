class Item:
  id: str
  name: str
  district: int

  def __init__(self, id: str, name: str, district: int) -> None:
    self.id = id
    self.name = name
    self.district = district

  def __repr__(self) -> str:
    return f"Item({self.id}[{self.name}]@{self.district})"

class TravelTime:
  from_id: str
  to_id: str
  duration: int

  def __init__(self, from_id: str, to_id: str, duration: int) -> None:
    self.from_id = from_id
    self.to_id = to_id
    self.duration = duration
  
  def __repr__(self) -> str:
    return f"TravelTime({self.from_id} -> {self.to_id}: {self.duration}s)"
  
  