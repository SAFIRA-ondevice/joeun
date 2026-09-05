import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"raspberry_pi"))
from decision_logic import decide

class FakeFallback:
    def __init__(self,recognized=True): self.recognized=recognized
    def classify(self,*_): return {"recognized":self.recognized,"label":"Siren" if self.recognized else "UNKNOWN","confidence":.7,"top":[]}

classes=["speech","drone","gunshot"]; thresholds={c:.5 for c in classes}
assert decide(classes,[.8,.7,.1],thresholds)["final"]=="SPEECH + DRONE"
assert decide(classes,[.1,.2,.1],thresholds,FakeFallback())["final"]=="추정: Siren"
assert decide(classes,[.1,.2,.1],thresholds,FakeFallback(False))["final"]=="UNKNOWN"
print("multilabel decision tests passed")
