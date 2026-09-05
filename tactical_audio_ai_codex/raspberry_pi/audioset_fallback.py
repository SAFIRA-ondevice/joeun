"""Optional AudioSet/YAMNet fallback. Imported only when configured."""
import csv
from pathlib import Path
import numpy as np

class YamnetFallback:
    def __init__(self,model_handle="https://tfhub.dev/google/yamnet/1",class_map=None,confidence=.25,margin=.05):
        try:
            import tensorflow_hub as hub
        except ImportError as e: raise RuntimeError("Install tensorflow and tensorflow-hub for YAMNet fallback") from e
        self.model=hub.load(model_handle); self.confidence=confidence; self.margin=margin
        if class_map:
            with Path(class_map).open(encoding="utf-8") as f: self.names=[r["display_name"] for r in csv.DictReader(f)]
        else:
            class_path=self.model.class_map_path().numpy().decode()
            with open(class_path,encoding="utf-8") as f: self.names=[r["display_name"] for r in csv.DictReader(f)]
    def classify(self,waveform,sample_rate=16000,top_k=3):
        if sample_rate!=16000: raise ValueError("YAMNet expects 16 kHz waveform")
        x=np.asarray(waveform,dtype=np.float32); x=x/(32768.0 if np.max(np.abs(x))>1.5 else 1.0)
        scores,_,_=self.model(x); mean=np.asarray(scores).mean(0); order=np.argsort(mean)[::-1][:top_k]
        guesses=[{"label":self.names[i],"score":float(mean[i])} for i in order]; top=guesses[0]
        recognized=top["score"]>=self.confidence and (len(guesses)==1 or top["score"]-guesses[1]["score"]>=self.margin)
        return {"recognized":recognized,"label":top["label"] if recognized else "UNKNOWN","confidence":top["score"],"top":guesses}
