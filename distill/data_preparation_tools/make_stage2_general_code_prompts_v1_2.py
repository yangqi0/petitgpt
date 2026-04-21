#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import random

OUT_DIR = "dataset/stage2/prompts_v1_2"
os.makedirs(OUT_DIR, exist_ok=True)

rng = random.Random(1234)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================================================
# General prompts v1.2
# More focused: email / summary / rewrite / explain
# Keep checklist small
# =========================================================
general_rows = []
gid = 1

# -------------------------
# A. Email
# -------------------------
recipients = [
    "a recruiter",
    "a professor",
    "a hiring manager",
    "a teammate",
    "a project supervisor",
    "a course instructor",
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
]

tones = [
    "short polite",
    "concise professional",
    "friendly but professional",
]

for recipient in recipients:
    for topic in topics:
        for tone in tones:
            prompt = f"Write a {tone} email to {recipient} asking for {topic}."
            general_rows.append(
                {
                    "id": f"general_email_{gid:04d}",
                    "task_type": "general_email",
                    "prompt": prompt,
                }
            )
            gid += 1

# -------------------------
# B. Summary
# -------------------------
summary_texts = [
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
]

summary_templates = [
    "Summarize this in 3 bullet points: '{text}'",
    "Give 3 key takeaways from this text: '{text}'",
    "Summarize this in one short paragraph: '{text}'",
]

for text in summary_texts:
    for tmpl in summary_templates:
        general_rows.append(
            {
                "id": f"general_summary_{gid:04d}",
                "task_type": "general_summary",
                "prompt": tmpl.format(text=text),
            }
        )
        gid += 1

# -------------------------
# C. Rewrite
# -------------------------
rewrite_inputs = [
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
]

rewrite_templates = [
    "Rewrite this to sound more polite: '{x}'",
    "Rewrite this to sound more professional: '{x}'",
    "Rewrite this to be shorter and clearer: '{x}'",
    "Rewrite this to sound friendlier while staying professional: '{x}'",
]

for x in rewrite_inputs:
    for tmpl in rewrite_templates:
        general_rows.append(
            {
                "id": f"general_rewrite_{gid:04d}",
                "task_type": "general_rewrite",
                "prompt": tmpl.format(x=x),
            }
        )
        gid += 1

# -------------------------
# D. Explain
# -------------------------
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
]

explain_templates = [
    "Explain {c} in simple terms, using at most 5 sentences.",
    "Explain {c} in simple language, using at most 5 sentences.",
]

for c in concepts:
    for tmpl in explain_templates:
        general_rows.append(
            {
                "id": f"general_explain_{gid:04d}",
                "task_type": "general_explain",
                "prompt": tmpl.format(c=c),
            }
        )
        gid += 1

# -------------------------
# E. Small amount of checklist / pros-cons
# -------------------------
check_topics = [
    "preparing for a Python coding interview",
    "cleaning a small CSV dataset",
    "debugging a function that returns the wrong result",
    "writing cleaner Python code",
    "organizing a small coding project",
    "reviewing a pull request",
    "preparing a small project for GitHub",
]

for topic in check_topics:
    general_rows.append(
        {
            "id": f"general_checklist_{gid:04d}",
            "task_type": "general_checklist",
            "prompt": f"Give me a 5-step checklist for {topic}.",
        }
    )
    gid += 1

pros_cons_topics = [
    "using Python for data analysis",
    "using unit tests in a small project",
    "using dictionaries for counting items",
    "working in Jupyter notebooks",
]

for topic in pros_cons_topics:
    general_rows.append(
        {
            "id": f"general_checklist_{gid:04d}",
            "task_type": "general_checklist",
            "prompt": f"List 3 pros and 3 cons of {topic}.",
        }
    )
    gid += 1

rng.shuffle(general_rows)
general_rows = general_rows[:220]


# =========================================================
# Code prompts v1.2
# Expand function-writing and bug-fixing heavily
# Keep script tasks moderate
# Keep explain+code small
# =========================================================
code_rows = []
cid = 1

# -------------------------
# A. Code write
# -------------------------
write_prompts = [
    "Write a Python function fib(n) that returns the n-th Fibonacci number iteratively. Raise ValueError if n < 0.",
    "Write a Python function factorial(n) that returns n!. Raise ValueError if n < 0.",
    "Write a Python function is_palindrome(s) that returns True if the string is a palindrome and False otherwise. Ignore case.",
    "Write a Python function reverse_string(s) that returns the reversed string.",
    "Write a Python function reverse_words(text) that reverses the order of words in a string.",
    "Write a Python function dedup_preserve_order(items) that removes duplicates while preserving the first occurrence order.",
    "Write a Python function intersection_preserve_order(a, b) that returns common elements while preserving the order from a.",
    "Write a Python function count_words(text) that returns a dictionary of lowercase word frequencies. Ignore commas and periods.",
    "Write a Python function count_chars(text) that returns a dictionary of character frequencies.",
    "Write a Python function count_lines(text) that returns the number of lines in a multi-line string.",
    "Write a Python function count_vowels(text) that counts the vowels in a string.",
    "Write a Python function all_unique(items) that returns True if all items are distinct.",
    "Write a Python function binary_search(nums, target) that returns the index of target in a sorted list, or -1 if not found.",
    "Write a Python function linear_search(nums, target) that returns the index of the first occurrence of target, or -1 if not found.",
    "Write a Python function gcd(a, b) that returns the greatest common divisor of two integers.",
    "Write a Python function lcm(a, b) that returns the least common multiple of two integers.",
    "Write a Python function flatten_once(items) that flattens a list by one level only.",
    "Write a Python function chunk_list(items, size) that splits a list into chunks of length size.",
    "Write a Python function top_k(nums, k) that returns the k largest numbers in descending order.",
    "Write a Python function merge_sorted_lists(a, b) that merges two sorted lists into one sorted list.",
    "Write a Python function remove_none(items) that returns a new list without None values.",
    "Write a Python function remove_empty_strings(items) that returns a new list without empty strings.",
    "Write a Python function safe_divide(a, b) that returns a / b, but returns None if b == 0.",
    "Write a Python function parse_duration(s) that converts strings like '1h30m' into total minutes.",
    "Write a Python function parse_integers(text) that extracts all integers from a string and returns them as a list.",
    "Write a Python function slugify(text) that converts a string into a lowercase slug using hyphens.",
    "Write a Python function group_by_length(words) that groups words by their length in a dictionary.",
    "Write a Python function group_by_first_letter(words) that groups words by their first letter in a dictionary.",
    "Write a Python function unique_sorted(nums) that returns the sorted unique values from a list of integers.",
    "Write a Python function second_largest(nums) that returns the second largest distinct integer in a list. Raise ValueError if it does not exist.",
    "Write a Python function running_sum(nums) that returns a list of cumulative sums.",
    "Write a Python function running_max(nums) that returns the running maximum after each position.",
    "Write a Python function moving_average(nums, k) that returns the simple moving average for each window of size k.",
    "Write a Python function clamp(x, low, high) that restricts x to the interval [low, high].",
    "Write a Python function pairwise_sum(a, b) that returns elementwise sums for two equal-length lists.",
    "Write a Python function transpose_matrix(matrix) for a rectangular list of lists.",
    "Write a Python function first_non_repeating_char(text) that returns the first non-repeating character, or None if it does not exist.",
    "Write a Python function find_duplicates(items) that returns the values that appear more than once, without duplicates in the output.",
    "Write a Python function word_lengths(words) that returns a dictionary mapping each word to its length.",
    "Write a Python function safe_get(d, key, default=None) that returns d[key] if present, otherwise default.",
    "Write a Python function lowercase_keys(d) that returns a new dictionary with all string keys lowercased.",
    "Write a Python function merge_counts(a, b) that merges two count dictionaries by summing values with the same key.",
    "Write a Python function filter_positive(nums) that returns only the positive numbers.",
    "Write a Python function sum_nested_lists(items) that returns the total sum of a list of lists of numbers.",
    "Write a Python function min_max_normalize(nums) that scales values to the range [0, 1]. If all values are equal, return a list of 0.0 values.",
]

write_styles = [
    "{prompt}",
    "{prompt} Return only code.",
]

for prompt in write_prompts:
    for tmpl in write_styles:
        code_rows.append(
            {
                "id": f"code_write_{cid:04d}",
                "task_type": "code_write",
                "prompt": tmpl.format(prompt=prompt),
            }
        )
        cid += 1

# -------------------------
# B. Code fix
# -------------------------
fix_prompts = [
    "Fix this function so it returns the reversed string correctly:\n\ndef reverse_string(s):\n    out = ''\n    for ch in s:\n        out = out + ch\n    return out\n",
    "Fix this function so it returns the factorial correctly:\n\ndef factorial(n):\n    result = 0\n    for i in range(1, n + 1):\n        result *= i\n    return result\n",
    "Fix this function so it counts words correctly:\n\ndef count_words(text):\n    words = text.lower().split()\n    counts = {}\n    for w in words:\n        counts[w] = 1\n    return counts\n",
    "Fix this function so it removes duplicates while keeping order:\n\ndef dedup_preserve_order(items):\n    seen = set()\n    out = []\n    for x in items:\n        if x in seen:\n            out.append(x)\n            seen.add(x)\n    return out\n",
    "Fix this binary search implementation:\n\ndef binary_search(nums, target):\n    left, right = 0, len(nums)\n    while left < right:\n        mid = (left + right) // 2\n        if nums[mid] == target:\n            return mid\n        elif nums[mid] < target:\n            right = mid - 1\n        else:\n            left = mid + 1\n    return -1\n",
    "Fix this function so it returns the sum of the list:\n\ndef list_sum(nums):\n    total = 0\n    for i in range(len(nums) + 1):\n        total += nums[i]\n    return total\n",
    "Fix this function so it returns only even numbers:\n\ndef get_evens(nums):\n    out = []\n    for x in nums:\n        if x % 2 == 1:\n            out.append(x)\n    return out\n",
    "Fix this function so it returns the maximum value:\n\ndef max_value(nums):\n    m = 0\n    for x in nums:\n        if x < m:\n            m = x\n    return m\n",
    "Fix this function so it checks palindromes correctly:\n\ndef is_palindrome(s):\n    return s == s\n",
    "Fix this function so it counts vowels correctly:\n\ndef count_vowels(text):\n    vowels = 'aeiou'\n    count = 0\n    for ch in text.lower():\n        if ch not in vowels:\n            count += 1\n    return count\n",
    "Fix this function so it finds the first matching index:\n\ndef linear_search(nums, target):\n    for i, x in enumerate(nums):\n        if x == target:\n            return x\n    return -1\n",
    "Fix this function so it computes the running sum correctly:\n\ndef running_sum(nums):\n    out = []\n    total = 0\n    for x in nums:\n        out.append(x)\n    return out\n",
    "Fix this function so it handles division by zero safely:\n\ndef safe_divide(a, b):\n    return a / b\n",
    "Fix this function so it groups words by length correctly:\n\ndef group_by_length(words):\n    result = {}\n    for w in words:\n        result[len(w)] = w\n    return result\n",
    "Fix this function so it returns the unique sorted values:\n\ndef unique_sorted(nums):\n    return sorted(nums)\n",
    "Fix this function so it computes the greatest common divisor correctly:\n\ndef gcd(a, b):\n    while b != 0:\n        a = b\n        b = a % b\n    return a\n",
    "Fix this function so it returns the running maximum correctly:\n\ndef running_max(nums):\n    out = []\n    current = 0\n    for x in nums:\n        if x < current:\n            current = x\n        out.append(current)\n    return out\n",
    "Fix this function so it lowercases dictionary keys correctly:\n\ndef lowercase_keys(d):\n    result = {}\n    for k, v in d.items():\n        result[k] = v.lower()\n    return result\n",
    "Fix this function so it counts lines correctly:\n\ndef count_lines(text):\n    return len(text)\n",
    "Fix this function so it removes None values correctly:\n\ndef remove_none(items):\n    return [x for x in items if x is None]\n",
    "Fix this function so it filters positive numbers correctly:\n\ndef filter_positive(nums):\n    return [x for x in nums if x < 0]\n",
]

fix_styles = [
    "{prompt}",
    "Fix the bug in this Python code and return only the corrected code:\n\n{prompt}",
]

for prompt in fix_prompts:
    for tmpl in fix_styles:
        code_rows.append(
            {
                "id": f"code_fix_{cid:04d}",
                "task_type": "code_fix",
                "prompt": tmpl.format(prompt=prompt),
            }
        )
        cid += 1

# -------------------------
# C. Code script / data processing
# -------------------------
script_prompts = [
    "Write a Python function read_numbers(path) that reads one integer per line from a text file and returns a list of integers.",
    "Write a Python function sum_csv_column(path, column_name) that reads a CSV file and returns the sum of one numeric column using the standard csv module.",
    "Write a Python function count_log_levels(lines) that counts how many lines contain INFO, WARNING, and ERROR.",
    "Write a Python function extract_emails(text) that returns a list of email addresses found in a string.",
    "Write a Python function extract_urls(text) that returns all URLs starting with http:// or https://.",
    "Write a Python function normalize_whitespace(text) that replaces repeated whitespace with a single space and strips the result.",
    "Write a Python function filter_rows_by_value(rows, key, value) where rows is a list of dictionaries. Return only rows where rows[i][key] == value.",
    "Write a Python function average_by_key(rows, key) where rows is a list of dictionaries and key refers to numeric values. Return the average.",
    "Write a Python function sort_people_by_age(people) where people is a list of dictionaries with keys name and age. Return the list sorted by age ascending.",
    "Write a Python function find_missing_numbers(nums) that returns the missing integers between the minimum and maximum values in a sorted list.",
    "Write a Python function line_lengths(text) that returns a list containing the length of each line in a multi-line string.",
    "Write a Python function most_common_word(text) that returns the most frequent lowercase word, ignoring commas and periods.",
    "Write a Python function parse_integers(text) that extracts all integers from a string and returns them as a list.",
    "Write a Python function csv_row_count(path) that returns the number of data rows in a CSV file using the standard csv module.",
]

script_styles = [
    "{prompt}",
    "{prompt} Return only a Python code block.",
]

for prompt in script_prompts:
    for tmpl in script_styles:
        code_rows.append(
            {
                "id": f"code_script_{cid:04d}",
                "task_type": "code_script",
                "prompt": tmpl.format(prompt=prompt),
            }
        )
        cid += 1

# -------------------------
# D. explain + code
# -------------------------
explain_prompts = [
    "In 2 short sentences, explain why a dictionary is useful for counting items. Then write a Python function count_words(text).",
    "In 3 short sentences, explain why binary search requires sorted input. Then write binary_search(nums, target).",
    "In 2 short sentences, explain the difference between a list and a set for duplicate handling. Then write dedup_preserve_order(items).",
    "In 2 short sentences, explain why iteration is often preferred over recursion for small utility functions. Then write fib(n) iteratively.",
    "In 3 short sentences, explain what a running sum is. Then write running_sum(nums).",
    "In 2 short sentences, explain why input validation matters. Then write safe_divide(a, b) that returns None on division by zero.",
    "In 2 short sentences, explain why normalizing text can help data processing. Then write normalize_whitespace(text).",
    "In 3 short sentences, explain why preserving order can matter when removing duplicates. Then write dedup_preserve_order(items).",
    "In 2 short sentences, explain why helper functions can improve readability. Then write chunk_list(items, size).",
    "In 2 short sentences, explain why descriptive variable names help readability. Then write running_max(nums).",
]

for prompt in explain_prompts:
    code_rows.append(
        {
            "id": f"code_explain_then_code_{cid:04d}",
            "task_type": "code_explain_then_code",
            "prompt": prompt,
        }
    )
    cid += 1

rng.shuffle(code_rows)
code_rows = code_rows[:360]


write_jsonl(os.path.join(OUT_DIR, "general_prompts_v1_2.jsonl"), general_rows)
write_jsonl(os.path.join(OUT_DIR, "code_prompts_v1_2.jsonl"), code_rows)

print("Wrote:")
print(os.path.join(OUT_DIR, "general_prompts_v1_2.jsonl"), len(general_rows))
print(os.path.join(OUT_DIR, "code_prompts_v1_2.jsonl"), len(code_rows))
