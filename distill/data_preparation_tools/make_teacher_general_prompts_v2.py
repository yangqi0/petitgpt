#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a large general-purpose teacher prompt set for synthetic SFT/distill.

Output format:
one JSON object per line with fields:
- id
- task_type
- prompt
- meta

Comments are in English by design.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
from typing import Any


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_email_rows() -> list[dict[str, Any]]:
    recipients = [
        "a recruiter",
        "a hiring manager",
        "a professor",
        "a teammate",
        "a project supervisor",
        "a course instructor",
        "a client",
        "a student advisor",
    ]
    topics = [
        "an update on a job application after an interview",
        "feedback on a submitted project",
        "clarification about an assignment deadline",
        "whether the role is still open",
        "rescheduling a meeting to next week",
        "confirmation that the documents were received",
        "a short extension for a project deadline",
        "a convenient time for a follow-up meeting",
        "the status of a bug report you submitted",
        "confirmation that your travel reimbursement form was received",
        "whether there are any next steps after a screening call",
        "whether a submitted code test has been reviewed",
    ]
    tones = [
        "short and polite",
        "concise and professional",
        "friendly but professional",
        "brief and respectful",
    ]
    rows = []
    i = 1
    for recipient, topic, tone in itertools.product(recipients, topics, tones):
        rows.append(
            {
                "id": f"general_email_{i:05d}",
                "task_type": "general_email",
                "prompt": f"Write a {tone} email to {recipient} asking for {topic}.",
                "meta": {"format": "email"},
            }
        )
        i += 1
    return rows


def build_summary_rows() -> list[dict[str, Any]]:
    texts = [
        "Python is widely used in data analysis, automation, and machine learning because it has a rich ecosystem of libraries and a readable syntax.",
        "Version control helps teams track code changes, collaborate safely, and recover earlier versions when mistakes happen.",
        "Unit tests improve reliability by checking expected behavior, catching regressions, and making refactoring safer.",
        "Data cleaning often includes removing duplicates, handling missing values, fixing inconsistent formats, and validating ranges.",
        "A neural network learns by adjusting parameters to reduce prediction error on training data.",
        "A hash table usually provides fast lookup, insert, and delete operations.",
        "A good technical project should define goals clearly, evaluate results systematically, and document trade-offs honestly.",
        "Gradient descent is an optimization algorithm that updates parameters in the direction that most reduces loss.",
        "Debugging often involves reproducing the issue, checking intermediate values, and narrowing down the source of the problem.",
        "APIs allow software systems to exchange data and functionality through defined interfaces.",
        "A CSV file stores tabular data as plain text, making it easy to inspect and exchange between tools.",
        "Functions make code easier to reuse, test, and organize.",
        "Indexes can speed up database queries by reducing the amount of data that must be scanned.",
        "Data leakage happens when information from outside the training split influences the model during training.",
        "Code review can catch bugs, improve readability, and spread knowledge across a team.",
        "Logging helps engineers understand what a system did before an error occurred.",
        "A stack is useful when the most recently added item should be processed first.",
        "A queue is useful when items should be processed in the order they arrive.",
        "Caching can reduce latency and repeated computation, but stale data must be handled carefully.",
        "Clear variable names and small functions make code easier to maintain.",
    ]
    templates = [
        ("general_summary", "Summarize this in 3 bullet points: '{text}'", {"bullet_count": 3}),
        ("general_summary", "Give 3 key takeaways from this text: '{text}'", {"bullet_count": 3}),
        ("general_summary", "Summarize this in one short paragraph: '{text}'", {"bullet_count": 0}),
        ("general_summary", "Rewrite this as 3 concise bullet points: '{text}'", {"bullet_count": 3}),
    ]
    rows = []
    i = 1
    for text, (task_type, tmpl, meta) in itertools.product(texts, templates):
        rows.append(
            {
                "id": f"{task_type}_{i:05d}",
                "task_type": task_type,
                "prompt": tmpl.format(text=text),
                "meta": meta,
            }
        )
        i += 1
    return rows


def build_rewrite_rows() -> list[dict[str, Any]]:
    inputs = [
        "Send me the file today.",
        "I want to know what is going on with my application.",
        "I am writing this email in order to ask whether you might perhaps have any updates regarding the interview process.",
        "Your report is late and needs changes.",
        "Can you fix this bug ASAP?",
        "Thank you very much for taking the time to read my message and consider my application.",
        "I do not understand your instructions.",
        "I think I may possibly be a decent fit for the role.",
        "This draft is confusing and too long.",
        "Please answer me quickly.",
        "The meeting time does not work for me.",
        "Your message was unclear.",
        "I need this by tomorrow.",
        "This explanation is hard to follow.",
        "The code is messy and difficult to review.",
        "I cannot attend at that time.",
        "This pull request needs major changes.",
        "I am still waiting for a response.",
        "The result looks wrong to me.",
        "This report repeats the same point too many times.",
    ]
    templates = [
        ("general_rewrite", "Rewrite this to sound more polite: '{x}'"),
        ("general_rewrite", "Rewrite this to sound more professional: '{x}'"),
        ("general_rewrite", "Rewrite this to be shorter and clearer: '{x}'"),
        ("general_rewrite", "Rewrite this to sound friendlier while staying professional: '{x}'"),
    ]
    rows = []
    i = 1
    for x, (task_type, tmpl) in itertools.product(inputs, templates):
        rows.append(
            {
                "id": f"{task_type}_{i:05d}",
                "task_type": task_type,
                "prompt": tmpl.format(x=x),
                "meta": {},
            }
        )
        i += 1
    return rows


def build_explain_rows() -> list[dict[str, Any]]:
    concepts = [
        "the difference between a Python list and a tuple",
        "the difference between a Python dictionary and a list",
        "the difference between a function and a method in Python",
        "when you would use a for loop instead of a while loop",
        "what a Python module is",
        "what a bug is in programming",
        "what debugging means",
        "what an API is",
        "the difference between a class and an object",
        "the difference between a library and a framework",
        "the difference between a variable and a constant",
        "what a test case is",
        "what version control is",
        "what a CSV file is",
        "what a function argument is",
        "the difference between a parameter and an argument",
        "why readable variable names matter",
        "what a loop does in Python",
        "why input validation matters",
        "what a stack and a queue are",
        "why binary search requires sorted input",
        "what overfitting means in machine learning",
        "what a tokenizer does in a language model pipeline",
        "why reproducibility matters in experiments",
    ]
    templates = [
        "Explain {c} in simple terms, using at most 5 sentences.",
        "Explain {c} in simple language, using at most 5 sentences.",
        "Explain {c} for a beginner, in no more than 5 sentences.",
    ]
    rows = []
    i = 1
    for c, tmpl in itertools.product(concepts, templates):
        rows.append(
            {
                "id": f"general_explain_{i:05d}",
                "task_type": "general_explain",
                "prompt": tmpl.format(c=c),
                "meta": {"max_sentences": 5},
            }
        )
        i += 1
    return rows


def build_checklist_rows() -> list[dict[str, Any]]:
    topics = [
        "preparing for a Python coding interview",
        "cleaning a small CSV dataset",
        "debugging a function that returns the wrong result",
        "writing cleaner Python code",
        "organizing a small coding project",
        "reviewing a pull request",
        "preparing a small project for GitHub",
        "writing a clear bug report",
        "running a small experiment and recording results",
        "checking a script before deployment",
        "understanding an unfamiliar codebase",
        "improving a technical résumé project",
    ]
    rows = []
    i = 1
    for topic in topics:
        rows.append(
            {
                "id": f"general_checklist_{i:05d}",
                "task_type": "general_checklist",
                "prompt": f"Give me a 5-step checklist for {topic}.",
                "meta": {"item_count": 5},
            }
        )
        i += 1
        rows.append(
            {
                "id": f"general_checklist_{i:05d}",
                "task_type": "general_checklist",
                "prompt": f"List 3 pros and 3 cons of {topic}.",
                "meta": {"item_count": 6},
            }
        )
        i += 1
    return rows


def build_commonsense_rows() -> list[dict[str, Any]]:
    rows = []
    i = 1
    prompts = [
        "A person has a laptop that will not turn on. Give 5 simple troubleshooting steps.",
        "Someone keeps missing deadlines because tasks are not organized. Suggest 5 practical improvements.",
        "A student wants to learn Python in three months. Give a realistic study plan in 5 steps.",
        "A small script runs but produces the wrong answer. Give 5 debugging actions in order.",
        "A teammate's message sounds too direct. Rewrite it to sound constructive and professional.",
        "A beginner confuses a list and a dictionary. Explain the difference in simple language.",
        "Give 4 short reasons why writing tests can save time later.",
        "Explain in 4 sentences why clear naming makes code easier to maintain.",
        "Summarize this in 3 bullet points: 'Good documentation makes projects easier to use, maintain, and extend.'",
        "Rewrite this to be shorter and clearer: 'I would like to inquire as to whether there may be any possibility of arranging another time for our discussion.'",
    ]
    for prompt in prompts:
        rows.append(
            {
                "id": f"general_misc_{i:05d}",
                "task_type": "general_checklist" if "steps" in prompt or "reasons" in prompt else "general_rewrite" if "Rewrite" in prompt else "general_summary" if "Summarize" in prompt else "general_explain",
                "prompt": prompt,
                "meta": {},
            }
        )
        i += 1
    return rows


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(build_email_rows())
    rows.extend(build_summary_rows())
    rows.extend(build_rewrite_rows())
    rows.extend(build_explain_rows())
    rows.extend(build_checklist_rows())
    rows.extend(build_commonsense_rows())
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_target", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = build_rows()
    rng.shuffle(rows)

    if args.n_target > 0:
        rows = rows[: min(args.n_target, len(rows))]

    write_jsonl(args.out, rows)
    print(f"Wrote {len(rows)} prompts to {args.out}")


if __name__ == "__main__":
    main()
