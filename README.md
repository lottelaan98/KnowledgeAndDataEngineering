# KnowledgeAndDataEngineering

*Research question*

Knowledge graph accuracy: How does integrating a medical knowledge graph (UMLS subset + symptom–disease dataset) improve the accuracy of automated diagnosis compared to baseline LLM or rule-based methods?

*Task division*
1. Data transformation + KG construction: Noortje
2. Build agent in python that uses LLM: Ripunjoy
3. Integrate KG in agent: Evan
4. Build UI where a patient can fill in his/her symptoms and the UI will show the concept extracted from the text given by the patient and predict the disease: Lotte
5. Evaluation: Lotte

*Literature and articles*

- Article: Leveraging Medical Knowledge Graphs Into Large Language Models for Diagnosis Prediction: Design and Application Study - https://arxiv.org/abs/2308.14321
- UMLS ontology: links different medical terms and coding systems so they can be used together for healthcare, research, and information retrieval.- https://www.nlm.nih.gov/research/umls/index.html
- Kaggle dataset: The dataset includes 24 different diseases, each with 50 symptom descriptions, resulting in a total of 1,200 data points. https://www.kaggle.com/datasets/niyarrbarman/symptom2disease. 

*Resources*

LLM guides and materials:
- LangChain/LangGraph courses: https://academy.langchain.com/collections. You need to make an account then you can follow the course for free. I recommend to do 'Quickstart: LangChain essentials - Python', and 'Quickstart: LangGraph essentials - Python'.
- LangChain Education Channel: https://www.youtube.com/@LangChain
- OpenAI Practical Guide to Building Agents: https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf
- Prompt Engineering and LLMs Guides: https://www.promptingguide.ai/research/llm-agents 


