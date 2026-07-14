# Annotation Guidelines for Dataset 4

**Version:** 1.0
**Purpose:** Emotion annotation of text data for the study dataset
**Prepared by:** Kian Tohidi
**Study:** A Comparative Evaluation of Large Language Models for Persian Sentiment Analysis and Emotion Detection in Social Media Texts

## Overview

Each text must be assigned one and only one emotion label. Annotators should base their decisions on the overall emotion expressed in the text, considering its literal meaning and immediate linguistic context: `anger`, `sadness`, `hatred`, `wonder`, or `fear`.

## Label Definitions

**Anger**: The text expresses irritation, outrage, or a clear sense of being upset or annoyed at someone or something.

Example: واقعاً عصبانی‌ام از این همه بی‌مسئولیتی.
(I am really angry about all this irresponsibility.)

**Sadness**: The text expresses sorrow, disappointment, grief, or a clear sense of being down or hurt.

Example: خیلی دلم گرفته، امروز روز سختی بود.
(I feel really down. Today was a hard day.)

**Hatred**: The text expresses strong hostility, contempt, or rejection directed at a person, group, or thing. Hatred goes beyond ordinary anger, it carries a clear wish to reject, exclude, or condemn the target.

Example: از این آدم و افکارش متنفرم، هیچ‌وقت نمی‌بخشمش.
(I hate this person and their ideas. I will never forgive them.)

**Wonder**: The text expresses astonishment, awe, or a clear reaction to something unexpected or amazing, whether the underlying event is good or bad.

Example: باورم نمی‌شه، اصلاً فکرشو نمی‌کردم همچین چیزی بشه.
(I can't believe it. I never expected something like this to happen.)

**Fear**: The text expresses worry, anxiety, or a clear sense of being scared or threatened by something.

Example: می‌ترسم فردا نتیجه‌ی آزمایش بد باشه.
(I am scared that tomorrow's test result will be bad.)

## Borderline Cases

**Mixed emotions**
If more than one emotion appears in the same text, label according to the dominant emotion. If neither dominates, choose the label that best represents the overall emotion expressed by the author.

**Anger vs. Hatred**
If the text shows irritation or annoyance without a clear wish to reject or condemn the target, label it `anger`. If the text shows strong hostility or a clear wish to reject, exclude, or condemn the target, label it `hatred`.

**Sarcasm**
Annotate according to the intended emotion whenever it is reasonably clear, not the literal words.

Example: آفرین! باز هم اینترنت قطع شد.
(Well done! The internet is down again.)
Label: `anger`

**Questions**
Questions that express frustration are labeled `anger`.

Example: چرا باز اینترنت قطع شد؟
(Why is the internet down again?)
Label: `anger`

Questions that express worry are labeled `fear`.

Example: نکنه فردا هم بارون بیاد و برنامه‌مون به‌هم بخوره؟
(What if it rains again tomorrow and ruins our plans?)
Label: `fear`

**Emojis**
Interpret emojis together with the surrounding text.

😢 → `sadness`
😡 → `anger`
😱 → `fear`
😲 → `wonder`

unless the surrounding text clearly suggests otherwise.

**Ambiguous texts**
If no emotion clearly dominates, choose the label that best represents the overall emotion expressed by the author.

## General Notes

* Choose only one label for each text.
* If a text feels hard to label, choose the label that best represents the overall emotion expressed by the author.
* If you are still unsure after that, choose the emotion that is most strongly hinted at, even if it is not stated directly.

## Annotation Procedure

1. Annotators receive identical text samples.
2. Original dataset labels are hidden.
3. Annotators work independently.
4. Communication between annotators is not allowed during annotation.
5. Completed annotation files are submitted:
   * `d4_annotator1.csv`
   * `d4_annotator2.csv`
6. Inter-annotator agreement is calculated using Cohen's κ.
7. Only after κ is calculated are disagreements reviewed and, if necessary, resolved through discussion.

Annotators must not modify their labels after discussing the texts. Any discussion occurs only after the independent annotation phase has been completed.
