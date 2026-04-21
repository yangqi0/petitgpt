#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os

OUT_DIR = "dataset/stage2/prompts"
os.makedirs(OUT_DIR, exist_ok=True)

general_prompts = [
    (
        "general_email",
        "Write a short polite email asking for an update on a job application after an interview.",
    ),
    ("general_email", "Write a short thank-you email after an interview."),
    (
        "general_email",
        "Write a polite email asking to reschedule a meeting to next week.",
    ),
    (
        "general_email",
        "Write a short email asking a professor for clarification about an assignment deadline.",
    ),
    (
        "general_email",
        "Write a polite message asking a recruiter whether the role is still open.",
    ),
    (
        "general_email",
        "Write a short email confirming that I received the documents successfully.",
    ),
    (
        "general_email",
        "Write a concise email asking for feedback on a submitted project.",
    ),
    (
        "general_email",
        "Write a polite message declining a meeting invitation because of a schedule conflict.",
    ),
    (
        "general_summary",
        "Summarize this in 3 bullet points: 'Python is widely used in data analysis, automation, and machine learning because it has a rich ecosystem of libraries and a readable syntax.'",
    ),
    (
        "general_summary",
        "Summarize this in 3 bullet points: 'Version control helps teams track code changes, collaborate safely, and recover earlier versions when mistakes happen.'",
    ),
    (
        "general_summary",
        "Summarize this in one short paragraph: 'Unit tests improve reliability by checking expected behavior, catching regressions, and making refactoring safer.'",
    ),
    (
        "general_summary",
        "Give 3 key takeaways from this text: 'Data cleaning often includes removing duplicates, handling missing values, fixing inconsistent formats, and validating ranges.'",
    ),
    (
        "general_summary",
        "Summarize this in 3 bullet points: 'A neural network learns by adjusting parameters to reduce prediction error on training data.'",
    ),
    (
        "general_summary",
        "Summarize this in one short paragraph: 'A hash table usually provides fast lookup, insert, and delete operations.'",
    ),
    (
        "general_summary",
        "Turn this into 3 concise bullet points: 'A good technical project should define goals clearly, evaluate results systematically, and document trade-offs honestly.'",
    ),
    (
        "general_summary",
        "Summarize this in plain English: 'Gradient descent is an optimization algorithm that updates parameters in the direction that most reduces loss.'",
    ),
    ("general_rewrite", "Rewrite this to sound more polite: 'Send me the file today.'"),
    (
        "general_rewrite",
        "Rewrite this to sound more professional: 'I want to know what is going on with my application.'",
    ),
    (
        "general_rewrite",
        "Rewrite this to be shorter and clearer: 'I am writing this email in order to ask whether you might perhaps have any updates regarding the interview process.'",
    ),
    (
        "general_rewrite",
        "Rewrite this message to sound friendlier: 'Your report is late and needs changes.'",
    ),
    (
        "general_rewrite",
        "Rewrite this in a professional tone: 'Can you fix this bug ASAP?'",
    ),
    (
        "general_rewrite",
        "Rewrite this to be more concise: 'Thank you very much for taking the time to read my message and consider my application.'",
    ),
    (
        "general_rewrite",
        "Rewrite this politely: 'I do not understand your instructions.'",
    ),
    (
        "general_rewrite",
        "Rewrite this to sound more confident but still polite: 'I think I may possibly be a decent fit for the role.'",
    ),
    (
        "general_explain",
        "Explain the difference between a Python list and a tuple in simple terms, using at most 5 sentences.",
    ),
    (
        "general_explain",
        "Explain the difference between a Python dictionary and a list in simple terms, using at most 5 sentences.",
    ),
    (
        "general_explain",
        "Explain the difference between a function and a method in Python, using at most 5 sentences.",
    ),
    (
        "general_explain",
        "Explain when you would use a for loop instead of a while loop, using simple language.",
    ),
    ("general_explain", "Explain what a Python module is, in 4 sentences or fewer."),
    ("general_explain", "Explain what a bug is in programming, in simple terms."),
    ("general_explain", "Explain what debugging means, using a tiny example."),
    (
        "general_explain",
        "Explain what an API is, in simple terms, without using jargon.",
    ),
    (
        "general_checklist",
        "Give me a 5-step checklist for preparing for a Python coding interview.",
    ),
    (
        "general_checklist",
        "Give me a 5-step checklist for cleaning a small CSV dataset.",
    ),
    ("general_checklist", "List 3 pros and 3 cons of using Python for data analysis."),
    ("general_checklist", "Give me 4 practical tips for writing cleaner Python code."),
    (
        "general_checklist",
        "Give me a short checklist for debugging a function that returns the wrong result.",
    ),
    (
        "general_checklist",
        "List 3 common mistakes beginners make when learning Python, with brief explanations.",
    ),
    (
        "general_checklist",
        "Give me 5 short tips for making an email sound more professional.",
    ),
    (
        "general_checklist",
        "Give me 4 short tips for organizing a small coding project.",
    ),
]

code_prompts = [
    (
        "code_write",
        "Write a Python function fib(n) that returns the n-th Fibonacci number iteratively. Raise ValueError if n < 0.",
    ),
    (
        "code_write",
        "Write a Python function factorial(n) that returns n!. Raise ValueError if n < 0.",
    ),
    (
        "code_write",
        "Write a Python function is_palindrome(s) that returns True if the string is a palindrome and False otherwise. Ignore case.",
    ),
    (
        "code_write",
        "Write a Python function reverse_string(s) that returns the reversed string.",
    ),
    (
        "code_write",
        "Write a Python function dedup_preserve_order(items) that removes duplicates while preserving the first occurrence order.",
    ),
    (
        "code_write",
        "Write a Python function count_words(text) that returns a dictionary of lowercase word frequencies. Ignore commas and periods.",
    ),
    (
        "code_write",
        "Write a Python function binary_search(nums, target) that returns the index of target in a sorted list, or -1 if not found.",
    ),
    (
        "code_write",
        "Write a Python function linear_search(nums, target) that returns the index of the first occurrence of target, or -1 if not found.",
    ),
    (
        "code_write",
        "Write a Python function gcd(a, b) that returns the greatest common divisor of two integers.",
    ),
    (
        "code_write",
        "Write a Python function lcm(a, b) that returns the least common multiple of two integers.",
    ),
    (
        "code_write",
        "Write a Python function flatten_once(items) that flattens a list by one level only.",
    ),
    (
        "code_write",
        "Write a Python function chunk_list(items, size) that splits a list into chunks of length size.",
    ),
    (
        "code_write",
        "Write a Python function top_k(nums, k) that returns the k largest numbers in descending order.",
    ),
    (
        "code_write",
        "Write a Python function merge_sorted_lists(a, b) that merges two sorted lists into one sorted list.",
    ),
    (
        "code_write",
        "Write a Python function remove_none(items) that returns a new list without None values.",
    ),
    (
        "code_write",
        "Write a Python function safe_divide(a, b) that returns a / b, but returns None if b == 0.",
    ),
    (
        "code_write",
        "Write a Python function parse_duration(s) that converts strings like '1h30m' into total minutes.",
    ),
    (
        "code_write",
        "Write a Python function slugify(text) that converts a string into a lowercase slug using hyphens.",
    ),
    (
        "code_write",
        "Write a Python function group_by_length(words) that groups words by their length in a dictionary.",
    ),
    (
        "code_write",
        "Write a Python function count_vowels(text) that counts the vowels in a string.",
    ),
    (
        "code_write",
        "Write a Python function unique_sorted(nums) that returns the sorted unique values from a list of integers.",
    ),
    (
        "code_write",
        "Write a Python function second_largest(nums) that returns the second largest distinct integer in a list. Raise ValueError if it does not exist.",
    ),
    (
        "code_write",
        "Write a Python function intersection_preserve_order(a, b) that returns common elements while preserving the order from a.",
    ),
    (
        "code_write",
        "Write a Python function running_sum(nums) that returns a list of cumulative sums.",
    ),
    (
        "code_fix",
        "Fix this Python function so it returns the reversed string correctly:\n\ndef reverse_string(s):\n    out = ''\n    for ch in s:\n        out = out + ch\n    return out\n",
    ),
    (
        "code_fix",
        "Fix this Python function so it returns the factorial correctly:\n\ndef factorial(n):\n    result = 0\n    for i in range(1, n + 1):\n        result *= i\n    return result\n",
    ),
    (
        "code_fix",
        "Fix this Python function so it counts words correctly:\n\ndef count_words(text):\n    words = text.lower().split()\n    counts = {}\n    for w in words:\n        counts[w] = 1\n    return counts\n",
    ),
    (
        "code_fix",
        "Fix this Python function so it removes duplicates while keeping order:\n\ndef dedup_preserve_order(items):\n    seen = set()\n    out = []\n    for x in items:\n        if x in seen:\n            out.append(x)\n            seen.add(x)\n    return out\n",
    ),
    (
        "code_fix",
        "Fix this binary search implementation:\n\ndef binary_search(nums, target):\n    left, right = 0, len(nums)\n    while left < right:\n        mid = (left + right) // 2\n        if nums[mid] == target:\n            return mid\n        elif nums[mid] < target:\n            right = mid - 1\n        else:\n            left = mid + 1\n    return -1\n",
    ),
    (
        "code_fix",
        "Fix this function so it returns the sum of the list:\n\ndef list_sum(nums):\n    total = 0\n    for i in range(len(nums) + 1):\n        total += nums[i]\n    return total\n",
    ),
    (
        "code_fix",
        "Fix this function so it returns only even numbers:\n\ndef get_evens(nums):\n    out = []\n    for x in nums:\n        if x % 2 == 1:\n            out.append(x)\n    return out\n",
    ),
    (
        "code_fix",
        "Fix this function so it returns the maximum value:\n\ndef max_value(nums):\n    m = 0\n    for x in nums:\n        if x < m:\n            m = x\n    return m\n",
    ),
    (
        "code_fix",
        "Fix this function so it checks palindromes correctly:\n\ndef is_palindrome(s):\n    return s == s\n",
    ),
    (
        "code_fix",
        "Fix this function so it counts vowels correctly:\n\ndef count_vowels(text):\n    vowels = 'aeiou'\n    count = 0\n    for ch in text.lower():\n        if ch not in vowels:\n            count += 1\n    return count\n",
    ),
    (
        "code_fix",
        "Fix this function so it finds the first matching index:\n\ndef linear_search(nums, target):\n    for i, x in enumerate(nums):\n        if x == target:\n            return x\n    return -1\n",
    ),
    (
        "code_fix",
        "Fix this function so it computes the running sum correctly:\n\ndef running_sum(nums):\n    out = []\n    total = 0\n    for x in nums:\n        out.append(x)\n    return out\n",
    ),
    (
        "code_fix",
        "Fix this function so it handles division by zero safely:\n\ndef safe_divide(a, b):\n    return a / b\n",
    ),
    (
        "code_fix",
        "Fix this function so it groups words by length correctly:\n\ndef group_by_length(words):\n    result = {}\n    for w in words:\n        result[len(w)] = w\n    return result\n",
    ),
    (
        "code_fix",
        "Fix this function so it returns the unique sorted values:\n\ndef unique_sorted(nums):\n    return sorted(nums)\n",
    ),
    (
        "code_fix",
        "Fix this function so it computes the greatest common divisor correctly:\n\ndef gcd(a, b):\n    while b != 0:\n        a = b\n        b = a % b\n    return a\n",
    ),
    (
        "code_script",
        "Write a Python function read_numbers(path) that reads one integer per line from a text file and returns a list of integers.",
    ),
    (
        "code_script",
        "Write a Python function sum_csv_column(path, column_name) that reads a CSV file and returns the sum of one numeric column using the standard csv module.",
    ),
    (
        "code_script",
        "Write a Python function count_log_levels(lines) that counts how many lines contain INFO, WARNING, and ERROR.",
    ),
    (
        "code_script",
        "Write a Python function extract_emails(text) that returns a list of email addresses found in a string.",
    ),
    (
        "code_script",
        "Write a Python function extract_urls(text) that returns all URLs starting with http:// or https://.",
    ),
    (
        "code_script",
        "Write a Python function normalize_whitespace(text) that replaces repeated whitespace with a single space and strips the result.",
    ),
    (
        "code_script",
        "Write a Python function filter_rows_by_value(rows, key, value) where rows is a list of dictionaries. Return only rows where rows[i][key] == value.",
    ),
    (
        "code_script",
        "Write a Python function average_by_key(rows, key) where rows is a list of dictionaries and key refers to numeric values. Return the average.",
    ),
    (
        "code_script",
        "Write a Python function sort_people_by_age(people) where people is a list of dictionaries with keys name and age. Return the list sorted by age ascending.",
    ),
    (
        "code_script",
        "Write a Python function find_missing_numbers(nums) that returns the missing integers between the minimum and maximum values in a sorted list.",
    ),
    (
        "code_script",
        "Write a Python function line_lengths(text) that returns a list containing the length of each line in a multi-line string.",
    ),
    (
        "code_script",
        "Write a Python function most_common_word(text) that returns the most frequent lowercase word, ignoring commas and periods.",
    ),
    (
        "code_explain_then_code",
        "In 2 short sentences, explain why a dictionary is useful for counting items. Then write a Python function count_words(text).",
    ),
    (
        "code_explain_then_code",
        "In 3 short sentences, explain why binary search requires sorted input. Then write binary_search(nums, target).",
    ),
    (
        "code_explain_then_code",
        "In 2 short sentences, explain the difference between a list and a set for duplicate handling. Then write dedup_preserve_order(items).",
    ),
    (
        "code_explain_then_code",
        "In 2 short sentences, explain why iteration is often preferred over recursion for small utility functions. Then write fib(n) iteratively.",
    ),
    (
        "code_explain_then_code",
        "In 3 short sentences, explain what a running sum is. Then write running_sum(nums).",
    ),
    (
        "code_explain_then_code",
        "In 2 short sentences, explain why input validation matters. Then write safe_divide(a, b) that returns None on division by zero.",
    ),
    (
        "code_explain_then_code",
        "In 2 short sentences, explain why normalizing text can help data processing. Then write normalize_whitespace(text).",
    ),
    (
        "code_explain_then_code",
        "In 3 short sentences, explain why preserving order can matter when removing duplicates. Then write dedup_preserve_order(items).",
    ),
]

basic_math_prompts = [
    (
        "math_word",
        "A notebook costs 8 euros. You buy 3 notebooks and get a 25% discount on the total. What is the final price? Show the steps briefly.",
    ),
    (
        "math_word",
        "A ladder is 12 meters long. Tom climbs it 5 times. How many meters does he climb in total? Show the steps briefly.",
    ),
    (
        "math_word",
        "A train travels 60 kilometers per hour for 3 hours. How far does it travel? Show the steps briefly.",
    ),
    (
        "math_word",
        "Anna buys 4 apples at 2 euros each and 3 bananas at 1 euro each. What is the total cost? Show the steps briefly.",
    ),
    (
        "math_word",
        "A box contains 24 pencils. If 6 students share them equally, how many pencils does each student get? Show the steps briefly.",
    ),
    (
        "math_word",
        "A shirt costs 40 euros. It is discounted by 15%. What is the sale price? Show the steps briefly.",
    ),
    (
        "math_word",
        "Mike reads 18 pages per day for 7 days. How many pages does he read in total? Show the steps briefly.",
    ),
    (
        "math_word",
        "A recipe needs 3 eggs for 1 cake. How many eggs are needed for 5 cakes? Show the steps briefly.",
    ),
    ("math_algebra", "Solve for x: 3x + 5 = 23. Show the steps briefly."),
    ("math_algebra", "Solve for x: 2x - 7 = 11. Show the steps briefly."),
    ("math_algebra", "Solve for x: 5x = 45. Show the steps briefly."),
    ("math_algebra", "Solve for x: x / 4 = 6. Show the steps briefly."),
    ("math_algebra", "Solve for x: 7x + 2 = 30. Show the steps briefly."),
    ("math_algebra", "If y = 2x + 3, what is y when x = 5? Show the steps briefly."),
    (
        "math_ratio",
        "A class has 20 students, and 5 of them are absent. What percentage of the class is absent? Show the steps briefly.",
    ),
    (
        "math_ratio",
        "A product costs 50 euros and its price increases by 10%. What is the new price? Show the steps briefly.",
    ),
    (
        "math_ratio",
        "In a group of 40 people, 30 prefer tea. What fraction and percentage prefer tea? Show the steps briefly.",
    ),
    (
        "math_ratio",
        "If 4 notebooks cost 12 euros, what is the cost of 7 notebooks at the same rate? Show the steps briefly.",
    ),
    (
        "math_ratio",
        "A car uses 6 liters of fuel to travel 100 kilometers. How many liters does it use to travel 250 kilometers? Show the steps briefly.",
    ),
    (
        "math_stats",
        "Given the list [2, 4, 4, 4, 5, 5, 7, 9], compute the mean and explain the steps briefly.",
    ),
    (
        "math_stats",
        "Given the list [2, 4, 4, 4, 5, 5, 7, 9], compute the median and explain the steps briefly.",
    ),
    (
        "math_stats",
        "Given the list [2, 4, 4, 4, 5, 5, 7, 9], compute the mode and explain the steps briefly.",
    ),
    (
        "math_stats",
        "Given the list [2, 4, 6, 8], compute the variance and explain the steps briefly.",
    ),
    (
        "math_stats",
        "Given the list [10, 12, 14, 16, 18], compute the mean and variance, and explain the steps briefly.",
    ),
]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for i, (task_type, prompt) in enumerate(rows, start=1):
            obj = {
                "id": f"{task_type}_{i:04d}",
                "task_type": task_type,
                "prompt": prompt,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


write_jsonl(os.path.join(OUT_DIR, "general_prompts_v1.jsonl"), general_prompts)
write_jsonl(os.path.join(OUT_DIR, "code_prompts_v1.jsonl"), code_prompts)
write_jsonl(os.path.join(OUT_DIR, "basic_math_prompts_v1.jsonl"), basic_math_prompts)

print("Wrote:")
print(os.path.join(OUT_DIR, "general_prompts_v1.jsonl"), len(general_prompts))
print(os.path.join(OUT_DIR, "code_prompts_v1.jsonl"), len(code_prompts))
print(os.path.join(OUT_DIR, "basic_math_prompts_v1.jsonl"), len(basic_math_prompts))
