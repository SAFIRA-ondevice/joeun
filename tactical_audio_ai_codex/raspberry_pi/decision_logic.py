def decide(classes, probabilities, thresholds, fallback=None, waveform=None, sample_rate=16000):
    percentages={c:round(float(v)*100,1) for c,v in zip(classes,probabilities)}
    active=[c for c,v in zip(classes,probabilities) if float(v)>=thresholds[c]]
    result={"targets":percentages,"detected":active,"fallback":None,"final":" + ".join(c.upper() for c in active) if active else None}
    if not active and fallback:
        fb=fallback.classify(waveform,sample_rate); result["fallback"]=fb
        result["final"]=(f"추정: {fb['label']}" if fb["recognized"] else "UNKNOWN")
    elif not active:
        result["final"]="UNKNOWN"
    return result
