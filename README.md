# KnowledgeAndDataEngineering

To run this python files please read requirements.txt file. 

**Clinical Decision-Support**

Clinical decision-support demo using an RDF/OWL knowledge graph + SPARQL reasoning + RAG explanations.

This project is a clinical decision-support system that helps users understand possible diseases based on the symptoms they describe in everyday language. The system extracts symptoms from the user’s text, matches them to medical concepts in a knowledge graph, and ranks the most likely diseases using clear and explainable scoring rules. It also determines how urgently the user should see a doctor by checking for alarm symptoms and summing symptom severity scores. Finally, it provides an easy-to-understand explanation for the results, grounded in verified medical information, while supporting clinical reasoning rather than replacing professional medical advice.

How does it work?
1. Takes user symptom text 
2. Extracts symptom candidates 
3. Maps to KG symptom IRIs 
4. Ranks diseases with explainable scoring 
5. Produces triage label via KG rules + optional text refinement
6. Optionally generates an explanation using local LLM (Llama 3.1 via Ollama) grounded in retrieved documents

*Some additional information*

**4. Ranks diseases with explainable scoring**

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

**5. Determines urgency with explainable triage rules**

In addition to disease ranking, the system determines how urgently the user should consult a doctor using rule-based triage queries over the RDF knowledge graph. The triage label can be established in two ways: with queries and using cosine similarity score.

Hard trigger query:
The system first checks whether any user-reported symptom is classified as an alarm symptom in the knowledge graph. This includes both categorical alarm symptoms and numeric alarm symptoms that exceed predefined thresholds (e.g. very high temperature or blood pressure). If any hard trigger is detected, the corresponding emergency triage label is immediately assigned and overrides all other reasoning.

Point-based triage scoring query:
If no hard trigger is activated, a triage score is computed by summing the scorePoints assigned to matched symptoms in the knowledge graph. This total score is then compared to a triage rule threshold.

Decision Rules:
- If ScoreSum ≥ Threshold → Urgent
- If 0 < ScoreSum < Threshold → Routine
- If ScoreSum = 0 → Unclear

Cosine similarity score:
This score is calculated using the see doctor recommendations, which are included in our knowledge base combined with the user’s input. A cosine-similarity comparison against 4 urgency labels to upgrade urgency when strong textual evidence is present. This is to prevent not recognising an urgent triage label.

Eventually the triage labels of both systems are combined to get the final label. 

