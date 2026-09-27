#!/usr/bin/env python3
from pathlib import Path
import time

request = Path(__file__).resolve().parent / ".reenroll_request"
request.write_text(str(time.time()), encoding="utf-8")
print("사용자 음성 재등록 요청을 전송했습니다.")
