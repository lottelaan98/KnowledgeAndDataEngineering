# KnowledgeAndDataEngineering

**Clinical Decision-Support**

Clinical decision-support demo using an RDF/OWL knowledge graph + SPARQL reasoning + RAG explanations.

How does it work?
1. Takes user symptom text 
2. Extracts symptom candidates 
3. Maps to KG symptom IRIs 
4. Ranks diseases with explainable scoring 
5. Produces triage label via KG rules + optional text refinement
6. Optionally generates an explanation using local LLM (Llama 3.1 via Ollama) grounded in retrieved documents



**4. Querying:**

As part of the final deliverable of the project we used two main queries based the final ttl file version: <br />
1)Information retrieval query that fetches relevant information on symptoms and diseases of interest. <br />
2)Disease scoring and ranking query.

Disease Scoring Logic: <br />
Diseases are ranked using a custom, interpretable scoring function implemented directly in SPARQL over the RDF knowledge graph. The score combines multiple clinically meaningful factors derived from the graph structure.

FinalScore = ( BaseScore × ImportanceCoeff ) + 0.1 ⋅ SymptomLocationBonus <br />
BaseScore = 0.6 ⋅ GlobalRarity + 0.2 ⋅ JaccardSimilarity + 0.2 ⋅ SymptomCoverage <br />

Brief Intuition:<br />
-JaccardSimilarity: measures overlap between user-reported symptoms and disease symptoms. <br />
-SymptomCoverage: measures how much of a disease’s symptom profile is explained by the user input. <br />
-GlobalRarity: rewards symptoms that occur in fewer diseases and are therefore more informative. <br />
-ImportanceCoeff: gives higher weight to primary symptoms compared to secondary ones. <br />
-SymptomLocationBonus: slightly favors diseases affecting the same body systems as the user’s symptoms. <br />

The formula is explicit, explainable and easy to extend with additional factors as more medical knowledge is added to the graph.

Note:<br />
A set of additional queries that were implemented but not used in the final deliverable can be found under: <br />
querying/ImplementedQueries/ <br />
This directory contains a Colab Νotebook with all the queries that were created and the corresponding ttl file (older version) used when developing them. 

