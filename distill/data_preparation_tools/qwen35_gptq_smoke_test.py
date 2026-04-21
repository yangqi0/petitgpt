#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from openai import OpenAI

BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
MODEL = os.environ.get("OPENAI_MODEL", "teacher")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

resp = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Write a short polite email asking for an update on a job application after an interview."},
    ],
    temperature=0.7,
    top_p=0.8,
    max_tokens=256,
    presence_penalty=1.5,
    extra_body={
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    },
)

print(resp.choices[0].message.content)
