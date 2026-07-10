# Annotation Guidelines for Dataset 2

**Version:** 1.0
**Date:** 2026-07-09
**Purpose:** Sentiment annotation of text data for the study dataset
**Prepared by:** [Your Name]
**Study:** [Study Name]

## Overview

Each text must be assigned one and only one sentiment label. Annotators should base their decisions on the overall sentiment expressed in the text, considering its literal meaning and immediate linguistic context: `Positive`, `Neutral`, or `Negative`.

## Label Definitions

**Positive**: The text says something good about someone or something. This includes clear praise, compliments, or a clear statement that the person is happy or satisfied.

Example: امروز واقعاً حس خوبی دارم، امیدوارم همه یه روز عالی داشته باشن.
(Today I feel really good. I hope everyone has a wonderful day.)

**Negative**: The text says something bad about someone or something. This includes insults, clear frustration, complaints, or a clear statement that the person is unhappy or dissatisfied.

Example: واقعاً خسته شدیم از این همه وعدهٔ بی‌نتیجه.
(We're really tired of all these empty promises.)

**Neutral**: The text does not show a clear opinion. This includes plain information, such as news headlines or factual questions about a product. It also includes texts that give both a positive and a negative view in a way that is balanced, with no side standing out.

Example: دیشب بازی ایران و ژاپن برگزار شد.
(Last night, the Iran vs. Japan match was held.)

## Borderline Cases

**Mixed sentiment**
If both positive and negative opinions appear in the same text, label according to the dominant sentiment. If neither dominates, choose `Neutral`.

**Sarcasm**
Annotate according to the intended sentiment whenever it is reasonably clear, not the literal words.

Example: آفرین! باز هم اینترنت قطع شد.
(Well done! The internet is down again.)
Label: `Negative`

**Questions**
Questions without any opinion are labeled `Neutral`.

Example: ساعت کاری فروشگاه چیه؟
(What are the store's working hours?)
Label: `Neutral`

Questions that express frustration are labeled `Negative`.

Example: چرا باز اینترنت قطع شد؟
(Why is the internet down again?)
Label: `Negative`

**Emojis**
Interpret emojis together with the surrounding text.

🙂 → `Positive`
😡 → `Negative`

unless the surrounding text clearly suggests otherwise.

**Ambiguous texts**
If no sentiment clearly dominates, choose `Neutral`.

## General Notes

* Choose only one label for each text.
* If a text feels hard to label, choose the label that best represents the overall sentiment expressed by the author.
* If you are still unsure after that, use `Neutral`.

## Annotation Procedure

1. Annotators receive identical text samples.
2. Original dataset labels are hidden.
3. Annotators work independently.
4. Communication between annotators is not allowed during annotation.
5. Completed annotation files are submitted:
   * `d1_annotator1.csv`
   * `d1_annotator2.csv`
6. Inter-annotator agreement is calculated using Cohen's κ.
7. Only after κ is calculated are disagreements reviewed and, if necessary, resolved through discussion.

Annotators must not modify their labels after discussing the texts. Any discussion occurs only after the independent annotation phase has been completed.
