from __future__ import annotations

import argparse
from typing import Any, Dict, List, Tuple

from general_utils import default_constraints_for_subfamily, write_jsonl

EMAIL = [
("job_followup_email", "Write a short polite email asking for an update on your job application after an interview last Tuesday. Keep it professional and under 90 words."),
("job_followup_email", "Write a short professional follow-up email after a first-round interview. Ask whether there are any updates on the hiring timeline. Keep it under 85 words."),
("job_followup_email", "Draft a polite email to ask if there has been any progress on your application since your interview last week. Keep it under 90 words."),
("job_followup_email", "Write a short email following up after an interview for a data analyst role. Ask for an update and thank the reader for their time. Keep it under 95 words."),
("job_followup_email", "Write a brief professional email asking whether there are any updates on your application after your recent interview. Keep it under 80 words."),
("job_followup_email", "Compose a short polite email to check on the status of your application after meeting with the team. Keep it under 90 words."),
("job_followup_email", "Write a concise professional email asking for an update after an interview earlier this month. Keep it under 85 words."),
("job_followup_email", "Write a short and polite email to follow up on an interview and ask whether there are any next steps to prepare for. Keep it under 95 words."),
("reschedule_email", "Write a short professional email asking to reschedule a meeting to next week. Keep it under 80 words."),
("reschedule_email", "Draft a polite email asking whether a meeting scheduled for Thursday can be moved to next week. Keep it under 85 words."),
("reschedule_email", "Write a brief professional email asking to move a meeting to Monday or Tuesday next week. Keep it under 90 words."),
("reschedule_email", "Compose a short email asking to reschedule tomorrow’s meeting because of a scheduling conflict. Keep it professional and under 85 words."),
("reschedule_email", "Write a short polite email asking whether a meeting can be moved to next Wednesday afternoon. Keep it under 85 words."),
("reschedule_email", "Draft a professional email asking to postpone a meeting until next week and inviting the other person to suggest a time. Keep it under 95 words."),
("reschedule_email", "Write a short email asking to reschedule a project meeting to next week due to an unexpected conflict. Keep it under 90 words."),
("reschedule_email", "Compose a brief professional note asking if the meeting can be moved from Friday to sometime next week. Keep it under 85 words."),
("thank_you_email", "Write a short thank-you email after a meeting. Keep it professional and under 75 words."),
("thank_you_email", "Draft a brief email thanking someone for taking the time to speak with you yesterday. Keep it warm and professional, under 80 words."),
("thank_you_email", "Write a short professional email thanking a recruiter for a helpful conversation. Keep it under 85 words."),
("thank_you_email", "Compose a brief thank-you email after a job interview. Keep it sincere and under 90 words."),
("thank_you_email", "Write a short email thanking a colleague for sending a document quickly. Keep it under 70 words."),
("thank_you_email", "Draft a concise professional message thanking someone for their feedback and support. Keep it under 75 words."),
("request_or_confirm_email", "Write a short polite email asking someone to send the latest version of a report. Keep it under 80 words."),
("request_or_confirm_email", "Draft a professional email asking whether someone could share the meeting notes. Keep it under 80 words."),
("request_or_confirm_email", "Write a brief polite email asking for a copy of the presentation slides. Keep it under 75 words."),
("request_or_confirm_email", "Compose a short professional email confirming that you received the file and thanking the sender. Keep it under 70 words."),
("request_or_confirm_email", "Write a short email confirming that you have attached the requested document. Keep it professional and under 70 words."),
("request_or_confirm_email", "Draft a polite email asking whether someone could resend an attachment that did not come through. Keep it under 85 words."),
("request_or_confirm_email", "Write a brief professional message confirming receipt of a report and saying you will review it today. Keep it under 75 words."),
("polite_short_message", "Write a short polite message asking a colleague whether they are free for a quick call this afternoon. Keep it under 40 words."),
("polite_short_message", "Write a brief professional message asking someone to confirm tomorrow’s meeting time. Keep it under 35 words."),
("polite_short_message", "Draft a short polite message asking a teammate to review a document when they have time. Keep it under 40 words."),
("polite_short_message", "Write a concise professional message apologizing for a delayed reply and saying you will respond fully tomorrow. Keep it under 45 words."),
("polite_short_message", "Compose a short polite message asking if a deadline can be extended by one day. Keep it under 40 words."),
("polite_short_message", "Write a brief professional message saying you are running five minutes late and will join the meeting soon. Keep it under 35 words."),
]

REWRITE = [
("rewrite_formal", "Rewrite this to be more formal: \"Thanks for the quick reply. I'll send the file tomorrow.\" Return only the rewritten text."),
("rewrite_formal", "Rewrite this to be more formal: \"Can we move the meeting to next week?\" Return only the rewritten text."),
("rewrite_formal", "Rewrite this to be more formal: \"Sorry for the late reply.\" Return only the rewritten text."),
("rewrite_formal", "Rewrite this to be more formal: \"I just wanted to check in about the job.\" Return only the rewritten text."),
("rewrite_formal", "Rewrite this to be more formal: \"Please send me the document when you can.\" Return only the rewritten text."),
("rewrite_formal", "Rewrite this to be more formal: \"Thanks again for your help.\" Return only the rewritten text."),
("rewrite_formal", "Rewrite this to be more formal: \"I got your email and will look at it today.\" Return only the rewritten text."),
("rewrite_formal", "Rewrite this to be more formal: \"I'm free after 3 p.m. if that works for you.\" Return only the rewritten text."),
("rewrite_formal", "Rewrite this to be more formal: \"Let me know what you think.\" Return only the rewritten text."),
("rewrite_formal", "Rewrite this to be more formal: \"I attached the updated version.\" Return only the rewritten text."),
("rewrite_concise", "Rewrite this to be more concise: \"I am writing this email in order to ask whether it would be possible to move our meeting to Friday afternoon.\" Return only the rewritten text."),
("rewrite_concise", "Rewrite this to be more concise: \"I wanted to reach out to see if you might have any updates regarding my application.\" Return only the rewritten text."),
("rewrite_concise", "Rewrite this to be more concise: \"Thank you very much for taking the time to review the document that I sent yesterday.\" Return only the rewritten text."),
("rewrite_concise", "Rewrite this to be more concise: \"I would be grateful if you could let me know whether the report is ready.\" Return only the rewritten text."),
("rewrite_concise", "Rewrite this to be more concise: \"I am sending this message to confirm that I received the file successfully.\" Return only the rewritten text."),
("rewrite_concise", "Rewrite this to be more concise: \"At your earliest convenience, could you please share the notes from the meeting?\" Return only the rewritten text."),
("rewrite_concise", "Rewrite this to be more concise: \"I am writing to follow up on our previous conversation from last week.\" Return only the rewritten text."),
("rewrite_concise", "Rewrite this to be more concise: \"Please let me know if there is any additional information that you would like from me.\" Return only the rewritten text."),
("rewrite_concise", "Rewrite this to be more concise: \"I wanted to thank you once again for your continued support and assistance.\" Return only the rewritten text."),
("rewrite_concise", "Rewrite this to be more concise: \"I hope you are doing well and having a good start to the week.\" Return only the rewritten text."),
("rewrite_polite", "Rewrite this to sound more polite: \"Send me the updated file today.\" Return only the rewritten text."),
("rewrite_polite", "Rewrite this to sound more polite: \"Can you reply today?\" Return only the rewritten text."),
("rewrite_polite", "Rewrite this to sound more polite: \"Move the meeting to next week.\" Return only the rewritten text."),
("rewrite_polite", "Rewrite this to sound more polite: \"I need the report now.\" Return only the rewritten text."),
("rewrite_polite", "Rewrite this to sound more polite: \"Check this and tell me what's wrong.\" Return only the rewritten text."),
("rewrite_polite", "Rewrite this to sound more polite: \"You forgot the attachment.\" Return only the rewritten text."),
("rewrite_polite", "Rewrite this to sound more polite: \"This explanation is unclear.\" Return only the rewritten text."),
("rewrite_polite", "Rewrite this to sound more polite: \"I can't meet tomorrow.\" Return only the rewritten text."),
("rewrite_clearer", "Rewrite this to be clearer: \"We should probably move things around a bit for next week.\" Return only the rewritten text."),
("rewrite_clearer", "Rewrite this to be clearer: \"I'm not sure the current version really works for what we need.\" Return only the rewritten text."),
("rewrite_clearer", "Rewrite this to be clearer: \"Maybe send it over later if that's easier.\" Return only the rewritten text."),
("rewrite_clearer", "Rewrite this to be clearer: \"The plan may need a few changes before we continue.\" Return only the rewritten text."),
("rewrite_clearer", "Rewrite this to be clearer: \"Let's talk about the issue when there's time.\" Return only the rewritten text."),
("rewrite_clearer", "Rewrite this to be clearer: \"The numbers don't quite line up yet.\" Return only the rewritten text."),
("rewrite_shorter", "Rewrite this to be shorter while keeping the meaning: \"Thank you for your message. I appreciate your quick response and will review the document tomorrow morning.\" Return only the rewritten text."),
("rewrite_shorter", "Rewrite this to be shorter while keeping the meaning: \"I wanted to let you know that I received the attachment and will send feedback later today.\" Return only the rewritten text."),
("rewrite_more_direct", "Rewrite this to be more direct: \"I was wondering whether you might be available at some point next week.\" Return only the rewritten text."),
("rewrite_more_direct", "Rewrite this to be more direct: \"I just wanted to ask if it would be okay to move the deadline by one day.\" Return only the rewritten text."),
("rewrite_more_direct", "Rewrite this to be more direct: \"Please feel free to let me know if you need anything else from me.\" Return only the rewritten text."),
("rewrite_shorter", "Rewrite this to be shorter while keeping the meaning: \"I am following up to see whether there have been any developments since our conversation.\" Return only the rewritten text."),
]

SUMMARY = [
("summary_3_bullets", "Summarize this in exactly 3 bullet points: \"Regular exercise can improve mood, support heart health, and help maintain energy levels.\" Keep each bullet short."),
("summary_3_bullets", "Summarize this in exactly 3 bullet points: \"Good sleep can improve focus, support mood, and help the body recover.\" Keep each bullet short."),
("summary_3_bullets", "Summarize this in exactly 3 bullet points: \"Drinking enough water helps the body function well, supports energy, and can improve concentration.\" Keep each bullet short."),
("summary_3_bullets", "Summarize this in exactly 3 bullet points: \"A clear weekly plan can reduce stress, improve focus, and make it easier to finish important tasks.\" Keep each bullet short."),
("summary_3_bullets", "Summarize this in exactly 3 bullet points: \"Taking notes during meetings helps people remember decisions, track action items, and reduce confusion later.\" Keep each bullet short."),
("summary_3_bullets", "Summarize this in exactly 3 bullet points: \"A budget helps you track spending, plan ahead, and avoid overspending.\" Keep each bullet short."),
("summary_3_bullets", "Summarize this in exactly 3 bullet points: \"Walking every day can improve fitness, reduce stress, and support better sleep.\" Keep each bullet short."),
("summary_3_bullets", "Summarize this in exactly 3 bullet points: \"Keeping files organized saves time, reduces mistakes, and makes teamwork easier.\" Keep each bullet short."),
("summary_3_bullets", "Summarize this in exactly 3 bullet points: \"Reading instructions carefully helps prevent errors, saves time, and improves the quality of the final result.\" Keep each bullet short."),
("summary_3_bullets", "Summarize this in exactly 3 bullet points: \"Regular breaks during work can help maintain focus, reduce fatigue, and improve productivity.\" Keep each bullet short."),
("summary_2_takeaways", "Give exactly 2 key takeaways from this text: \"Preparing before a meeting helps people ask better questions and use time more effectively.\" Keep each takeaway short."),
("summary_2_takeaways", "Give exactly 2 key takeaways from this text: \"Replying clearly in emails can reduce confusion and speed up decisions.\" Keep each takeaway short."),
("summary_2_takeaways", "Give exactly 2 key takeaways from this text: \"Saving a little money regularly can make future expenses easier to manage.\" Keep each takeaway short."),
("summary_2_takeaways", "Give exactly 2 key takeaways from this text: \"Reviewing your work before sending it can catch mistakes and improve clarity.\" Keep each takeaway short."),
("summary_2_takeaways", "Give exactly 2 key takeaways from this text: \"Setting priorities at the start of the day can help you focus on the most important tasks.\" Keep each takeaway short."),
("summary_2_takeaways", "Give exactly 2 key takeaways from this text: \"Clear deadlines make it easier to plan work and avoid last-minute stress.\" Keep each takeaway short."),
("summary_1_sentence", "Summarize this in one sentence: \"Exercise can improve mood, support heart health, and help maintain energy levels.\""),
("summary_1_sentence", "Summarize this in one sentence: \"A budget helps you plan spending and understand where your money goes.\""),
("summary_1_sentence", "Summarize this in one sentence: \"Meeting notes help teams remember decisions and follow up on next steps.\""),
("summary_1_sentence", "Summarize this in one sentence: \"Clear communication can reduce mistakes and make teamwork smoother.\""),
("summary_1_sentence", "Summarize this in one sentence: \"A simple daily plan can improve focus and reduce stress.\""),
("extract_key_points", "List exactly 3 key points from this text: \"Good file naming makes documents easier to find, reduces confusion, and helps teams stay organized.\" Keep each point short."),
("extract_key_points", "List exactly 3 main points from this text: \"Preparing a shopping list can save time, reduce impulse buying, and help control spending.\" Keep each point short."),
("extract_key_points", "Extract exactly 2 main ideas from this text: \"A follow-up email can show professionalism and help move a conversation forward.\" Keep each point short."),
("extract_key_points", "List exactly 3 key points from this text: \"Asking clear questions often leads to clearer answers and fewer misunderstandings.\" Keep each point short."),
]

EXPLAIN = [
("explain_budget_simple", "Explain in at most 3 sentences what a budget is, in simple everyday language."),
("explain_deadline_simple", "Explain in at most 3 sentences what a deadline is, in simple everyday language."),
("explain_password_manager_simple", "Explain in at most 3 sentences what a password manager is, in simple everyday language."),
("explain_meeting_agenda_simple", "Explain in at most 3 sentences what a meeting agenda is, in simple everyday language."),
("explain_reminder_simple", "Explain in at most 3 sentences what a reminder is, in simple everyday language."),
("explain_receipt_simple", "Explain in at most 3 sentences what a receipt is, in simple everyday language."),
("explain_schedule_simple", "Explain in at most 3 sentences what a schedule is, in simple everyday language."),
("explain_feedback_simple", "Explain in at most 3 sentences what feedback means in a work context, in simple everyday language."),
("explain_draft_simple", "Explain in at most 3 sentences what a draft is, in simple everyday language."),
("explain_subscription_simple", "Explain in at most 3 sentences what a subscription is, in simple everyday language."),
("explain_task_list_simple", "Explain in at most 3 sentences what a task list is, in simple everyday language."),
("explain_attachment_simple", "Explain in at most 3 sentences what an attachment is in email, in simple everyday language."),
("compare_debit_credit_simple", "Explain in simple terms the difference between a debit card and a credit card. Use at most 3 sentences."),
("compare_receipt_invoice_simple", "Explain in simple terms the difference between a receipt and an invoice. Use at most 3 sentences."),
("compare_note_report_simple", "Explain in simple terms the difference between a note and a report. Use at most 3 sentences."),
("compare_reminder_alarm_simple", "Explain in simple terms the difference between a reminder and an alarm. Use at most 3 sentences."),
("compare_agenda_notes_simple", "Explain in simple terms the difference between a meeting agenda and meeting notes. Use at most 3 sentences."),
("compare_password_pin_simple", "Explain in simple terms the difference between a password and a PIN. Use at most 3 sentences."),
("compare_goal_deadline_simple", "Explain in simple terms the difference between a goal and a deadline. Use at most 3 sentences."),
("compare_saving_spending_simple", "Explain in simple terms the difference between saving and spending. Use at most 3 sentences."),
]

def family_of(subfamily: str) -> str:
    if subfamily.endswith("email") or subfamily == "polite_short_message" or subfamily == "request_or_confirm_email":
        return "email_message"
    if subfamily.startswith("rewrite_"):
        return "rewrite_style"
    if subfamily.startswith("summary_") or subfamily == "extract_key_points":
        return "summary_bullets"
    return "explain_compare"

def build_records() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for prefix, items in [("e", EMAIL), ("r", REWRITE), ("s", SUMMARY), ("x", EXPLAIN)]:
        for idx, (subfamily, prompt) in enumerate(items, start=1):
            family = family_of(subfamily)
            rows.append({
                "id": f"tmpl_{prefix}{idx:02d}",
                "family": family,
                "subfamily": subfamily,
                "source": "template_seed",
                "canonical_prompt": prompt,
                "constraints": default_constraints_for_subfamily(subfamily),
                "meta": {
                    "seed_version": "v1",
                    "from_template": True,
                },
            })
    return rows

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()
    rows = build_records()
    write_jsonl(args.out_jsonl, rows)
    print(f"Wrote {len(rows)} seeds to {args.out_jsonl}")

if __name__ == "__main__":
    main()
