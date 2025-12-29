# Context-Aware Query Expansion & Topic Tagging

Most chat systems answer each question in isolation.

Real conversations don’t work like that.

Example:

Who is PM of India 

what about US

his duties?


Humans automatically track:

- we are talking about politics  
- “US” switches the country  
- “his” refers to the PM  
- the last question asks about duties  

Traditional rule systems break.  
LLMs often hallucinate.

This project builds a lightweight reasoning engine that:

- remembers conversational context
- predicts topic using ML
- rewrites vague queries into explicit ones
- answers using grounded knowledge
- asks for clarification when context is missing

---

## 🧠 Architecture

![App Screenshot](https://github.com/Shubhf/basic/blob/main/Flowchart.png)


### Components

**Dialogue Act Detector**  
Detects whether input is:

- new query
- contextual continuation
- casual chat

**Topic Classifier (ML)**  
TF-IDF + Logistic Regression trained on synthetic data:

- Politics
- Sports
- General
- Unknown

**Context State**  
Tracks conversation memory:

| Field   | Meaning |
|--------|---------|
| domain | politics, sports, general |
| subject | india, us, uk… |
| role | prime minister, captain… |
| intent | who, duties, info… |

**Expansion Engine**

Turns vague messages into explicit queries.

Example:


