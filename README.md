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


