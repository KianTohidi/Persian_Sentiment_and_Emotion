# Annotation Guidelines for Dataset 2

**Version:** 1.0
**Purpose:** Emotion annotation of text data for the study dataset
**Prepared by:** Kian Tohidi
**Study:** A Comparative Evaluation of Large Language Models for Persian Sentiment Analysis and Emotion Detection in Social Media Texts

## Overview

Each text must be assigned one and only one emotion label. Annotators should base their decisions on the overall emotion expressed in the text, considering its literal meaning and immediate linguistic context: `happiness`, `sadness`, `anger`, `intense emotions`, or `neutral`.

## Label Definitions

**Happiness**: The text expresses joy, excitement, or a clear sense of being pleased or glad about something.

Example: امروز واقعاً حس خوبی دارم، امیدوارم همه یه روز عالی داشته باشن.
(Today I feel really good. I hope everyone has a wonderful day.)

**Sadness**: The text expresses sorrow, disappointment, grief, or a clear sense of being down or hurt.

Example: خیلی دلم گرفته، امروز روز سختی بود.
(I feel really down. Today was a hard day.)

**Anger**: The text expresses irritation, outrage, or a clear sense of being upset or annoyed at someone or something.

Example: واقعاً عصبانی‌ام از این همه بی‌مسئولیتی.
(I am really angry about all this irresponsibility.)

**Intense Emotions**: The text expresses a strong emotion that goes beyond ordinary happiness, sadness, or anger, most often love, deep affection, or another emotion expressed with unusual intensity.

Example: بی‌نهایت دوستت دارم، تو بهترین اتفاق زندگیمی.
(I love you endlessly. You are the best thing that ever happened to me.)

**Neutral**: The text does not show a clear emotion. This includes plain information, such as news headlines or factual statements. It also includes texts that mix different emotions in a balanced way, with no single emotion standing out.

Example: دیشب بازی ایران و ژاپن برگزار شد.
(Last night, the Iran vs. Japan match was held.)

## Borderline Cases

**Mixed emotions**
If more than one emotion appears in the same text, label according to the dominant emotion. If neither dominates, choose `neutral`.

**Sarcasm**
Annotate according to the intended emotion whenever it is reasonably clear, not the literal words.

Example: آفرین! باز هم اینترنت قطع شد.
(Well done! The internet is down again.)
Label: `anger`

**Questions**
Questions without any clear emotion are labeled `neutral`.

Example: ساعت کاری فروشگاه چیه؟
(What are the store's working hours?)
Label: `neutral`

Questions that express frustration are labeled `anger`.

Example: چرا باز اینترنت قطع شد؟
(Why is the internet down again?)
Label: `anger`

**Emojis**
Interpret emojis together with the surrounding text.

🙂 → `happiness`
😢 → `sadness`
😡 → `anger`
❤️ → `intense emotions`

unless the surrounding text clearly suggests otherwise.

**Ambiguous texts**
If no emotion clearly dominates, choose `neutral`.

## General Notes

* Choose only one label for each text.
* If a text feels hard to label, choose the label that best represents the overall emotion expressed by the author.
* If you are still unsure after that, use `neutral`.

## Annotation Procedure

1. Annotators receive identical text samples.
2. Original dataset labels are hidden.
3. Annotators work independently.
4. Communication between annotators is not allowed during annotation.
5. Completed annotation files are submitted:
   * `d2_annotator1.csv`
   * `d2_annotator2.csv`
6. Inter-annotator agreement is calculated using Cohen's κ.
7. Only after κ is calculated are disagreements reviewed and, if necessary, resolved through discussion.

Annotators must not modify their labels after discussing the texts. Any discussion occurs only after the independent annotation phase has been completed.
