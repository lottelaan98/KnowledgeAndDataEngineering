# KnowledgeAndDataEngineering

*Research question*

Knowledge graph accuracy: How does integrating a medical knowledge graph (UMLS subset + symptom–disease dataset) improve the accuracy of automated diagnosis compared to baseline LLM or rule-based methods?

*Literature and articles*

- Article: Leveraging Medical Knowledge Graphs Into Large Language Models for Diagnosis Prediction: Design and Application Study - https://arxiv.org/abs/2308.14321
- UMLS ontology: links different medical terms and coding systems so they can be used together for healthcare, research, and information retrieval.- https://www.nlm.nih.gov/research/umls/index.html
- Kaggle dataset: The dataset includes 24 different diseases, each with 50 symptom descriptions, resulting in a total of 1,200 data points. https://www.kaggle.com/datasets/niyarrbarman/symptom2disease.
- Medical LLM leaderboard: https://huggingface.co/blog/leaderboard-medicalllm

*Resources*

LLM guides and materials:
- LangChain/LangGraph courses: https://academy.langchain.com/collections. You need to make an account then you can follow the course for free. I recommend to do 'Quickstart: LangChain essentials - Python', and 'Quickstart: LangGraph essentials - Python'.
- LangChain Education Channel: https://www.youtube.com/@LangChain
- OpenAI Practical Guide to Building Agents: https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf
- Prompt Engineering and LLMs Guides: https://www.promptingguide.ai/research/llm-agents

** Querying **
As part of the final deliverable of the project we used two main queries based the final ttl file version:
1)Information retrieval query that fetches relevant information on symptoms and diseases of interest.
2)Disease scoring and ranking query.

Disease Scoring Logic:
Diseases are ranked using a custom, interpretable scoring function implemented directly in SPARQL over the RDF knowledge graph. The score combines multiple clinically meaningful factors derived from the graph structure.

FinalScore = ( BaseScore × ImportanceCoeff ) + 0.1 ⋅ SymptomLocationBonus
BaseScore = 0.6 ⋅ GlobalRarity + 0.2 ⋅ JaccardSimilarity + 0.2 ⋅ SymptomCoverage

Brief Intuition:
-JaccardSimilarity: measures overlap between user-reported symptoms and disease symptoms.
-SymptomCoverage: measures how much of a disease’s symptom profile is explained by the user input.
-GlobalRarity: rewards symptoms that occur in fewer diseases and are therefore more informative.
-ImportanceCoeff: gives higher weight to primary symptoms compared to secondary ones.
-SymptomLocationBonus: slightly favors diseases affecting the same body systems as the user’s symptoms.

The formula is explicit, explainable and easy to extend with additional factors as more medical knowledge is added to the graph.

Note:
A set of additional queries that were implemented but not used in the final deliverable can be found under: 
querying/ImplementedQueries/ 
This directory contains a Colab notebook and the corresponding TTL version used when developing these queries.

