import re
import ollama

class ResultsSummarizer:
    def __init__(self):
        """
        Initializes the summarizer with Ollama model configuration.
        """
        self.model = "llama3.2"  # Can be replaced with other models as needed
        self.config = {
            "temperature": 0.5, #Control Randomness
            "max_tokens": 1000, # maximum number of tokens (words or sub-words) the model should generate in a single response.
            "top_p": 1.0, #leading to more coherent and relevant text.
             "top_k": 50, # Top-k sampling : Limits the sampling pool to the top K most likely next tokens.
            "frequency_penalty": 0.5, # Reduces repetition
        }

        # Load scientific spaCy model if available, fallback to standard model
        
    def extract_evidence_sentences(self, filtered_doc):
        """
        Extract unique sentences that contain evidence propositions from the filtered document,
        allowing for missing components.
        
        Args:
            filtered_doc (dict): The document containing study information
        
        Returns:
            list: A list of unique sentences containing evidence propositions
        """
        if not filtered_doc or not isinstance(filtered_doc, list) or len(filtered_doc) == 0:
            return []
        
        doc = filtered_doc[0]
        
        # Get all sentences
        sentences = doc.get('Sentence-level_breakdown.Text', [])
        
        # Get evidence proposition data
        interventions = doc.get('Sentence-level_breakdown.Evidence_Propositions.Intervention', [])
        observations = doc.get('Sentence-level_breakdown.Evidence_Propositions.Observation', [])
        outcomes = doc.get('Sentence-level_breakdown.Evidence_Propositions.Outcome', [])
        
        # Create a set to store unique evidence sentences
        unique_evidence_sentences = set()
        
        # Check if we have at least some proposition data
        if sentences and (interventions or observations or outcomes):
            # First, try to find sentences with all three components
            if interventions and observations and outcomes:
                for i in range(min(len(interventions), len(observations), len(outcomes))):
                    # Skip empty components
                    if not interventions[i] or not observations[i] or not outcomes[i]:
                        continue
                        
                    # Find the sentence containing this evidence proposition
                    for sentence in sentences:
                        # Check if all components are in this sentence
                        if (interventions[i] in sentence and 
                            observations[i] in sentence and 
                            outcomes[i] in sentence):
                            # Add the sentence to our set of unique sentences
                            unique_evidence_sentences.add(sentence)
                            break
            
            # If we couldn't find sentences with all components, look for sentences with at least two components
            if not unique_evidence_sentences and interventions and (observations or outcomes):
                for i in range(min(len(interventions), max(len(observations), len(outcomes)))):
                    # Skip if intervention is empty
                    if not interventions[i]:
                        continue
                        
                    # Find sentences with intervention and either observation or outcome
                    for sentence in sentences:
                        has_observation = i < len(observations) and observations[i] and observations[i] in sentence
                        has_outcome = i < len(outcomes) and outcomes[i] and outcomes[i] in sentence
                        
                        if interventions[i] in sentence and (has_observation or has_outcome):
                            unique_evidence_sentences.add(sentence)
                            break
        
        # Convert the set back to a list
        return list(unique_evidence_sentences)

    def find_sentences_with_all_entities_in_sections(self, text, entity_data):
        """
        Find sentences in the given text (Results & Conclusion) that contain all three entity types.
        """
        if not text:
            print(" No text provided for entity matching!")
            return []
        
        # Split text into sentences (simple split by period or newline)
        sentences = [s.strip() for s in re.split(r"(?<=\.)\s|\n", text) if s.strip()]
        intervention_terms = []
        outcome_terms = []
        observation_terms = []
        count_terms = []
        
        for sentence_data in entity_data.get('sentences', []):
            for prop in sentence_data.get('Evidence Propositions', []):
                if 'Intervention' in prop and prop['Intervention'] and prop['Intervention'] not in intervention_terms:
                    intervention_terms.append(prop['Intervention'].lower())
                if 'Outcome' in prop and prop['Outcome'] and prop['Outcome'] not in outcome_terms:
                    outcome_terms.append(prop['Outcome'].lower())
                if 'Observation' in prop and prop['Observation'] and prop['Observation'] not in observation_terms:
                    observation_terms.append(prop['Observation'].lower())
                if 'Count' in prop and prop['Count'] and prop['Count'] not in count_terms:
                    count_terms.append(prop['Count'].lower())
        
        print("\n Intervention Terms:", intervention_terms)
        print(" Outcome Terms:", outcome_terms)
        print(" Observation Terms:", observation_terms)
        print(" Count terms:", count_terms)

        sentences_with_all_entities = []

        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check for presence of all three entity types
            has_intervention = any(term in sentence_lower for term in intervention_terms)
            has_outcome = any(term in sentence_lower for term in outcome_terms)
            has_observation = any(term in sentence_lower for term in observation_terms)
            has_count = any(term in sentence_lower for term in count_terms)

            print(f"\n Checking Sentence: {sentence}")
            print(f"Intervention found: {has_intervention},  Outcome found: {has_outcome},  Observation found: {has_observation}, Count found : {has_count}")

            if (has_intervention and has_outcome and has_observation) or has_count:
                sentences_with_all_entities.append(sentence)

        if not sentences_with_all_entities:
            print(" No sentences matched all three entities!\n")

        return sentences_with_all_entities

    def clean_sentence(self, sentence):
        """
        Remove section headers like RESULTS:, CONCLUSIONS:, etc. from the sentence.
        """
        # Pattern to match section headers at the beginning of a sentence
        pattern = r"^(TITLE|RESULTS|CONCLUSIONS|OBJECTIVE|METHODS|BACKGROUND|INTRODUCTION|DISCUSSION|PURPOSE)\s*:\s*"
        cleaned = re.sub(pattern, "", sentence, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s*\[\d+\]\s*$', '', cleaned)
        cleaned = re.sub(r'\s*\[\d+\]\s*', ' ', cleaned)
        return cleaned.strip()
    
    def clean_summary_output(self, text):
        """
        Clean the summary output to remove bullet points, stars, and section headers.
        
        Args:
            text (str): The summary text to clean
            
        Returns:
            str: The cleaned summary text
        """
        # Remove bullet points and stars
        cleaned = re.sub(r'^[\*\•\-]\s*', '', text, flags=re.MULTILINE)
        
        # Remove section headers like "Key Findings:" or "Clinical Relevance:"
        cleaned = re.sub(r'\*\*(Key Findings|Clinical Relevance|Primary Intervention|Observed Outcomes|Study Overview|Significant conclusions|Study Significance)\*\*:\s*', '', cleaned)
        cleaned = re.sub(r'(Key Findings|Clinical Relevance|Primary Intervention|Observed Outcomes|Study Overview|Significant conclusions|Study Significance):\s*', '', cleaned)
        
        # Convert multiple newlines to spaces
        cleaned = re.sub(r'\n+', ' ', cleaned)
        
        # Remove any remaining markdown formatting
        cleaned = re.sub(r'\*\*|\*', '', cleaned)
        
        # Fix any double spaces
        cleaned = re.sub(r'\s{2,}', ' ', cleaned)
        
        return cleaned.strip()
    
    def generate_summary(self, summary_data, display_results, query, complete_data):
        """
        Generate summaries from the provided data
        
        Args:
            summary_data: Dictionary containing clean_data_combined
            display_results: List of document metadata
            query: The search query
            
        Returns:
            Dictionary with summary_parts 
        """
        clean_data_combined = summary_data.get('clean_data_combined', [])

        # Initialize results
        extracted_summaries = []
        summary_parts = []
        evidence_sentences_list = []

        # Process each document
        for i, (clean_data, doc_metadata) in enumerate(zip(clean_data_combined, display_results)):
            # Extract abstract text
            print("\n" + "="*80)
            print(f" Processing Document {i + 1}: {doc_metadata.get('title', 'Unknown Title')}")

            abstract = clean_data.get('abstract', '')
            print("\n Full Abstract Text:\n", abstract[:500], "..." if len(abstract) > 500 else "")

            key_sentences = self.find_sentences_with_all_entities_in_sections(abstract, clean_data)

            # Clean sentences by removing section headers
            cleaned_sentences = [self.clean_sentence(sentence) for sentence in key_sentences]
            extracted_summary = ""
            if cleaned_sentences:
                extracted_summary = " ".join(cleaned_sentences)
                print("\n Entity-based extraction result:", extracted_summary)
            
            extracted_summaries.append({
                'text': extracted_summary,
                'doc_id': doc_metadata.get('doc_id', ''),
                'index': i + 1
            })
            if complete_data:
                doc_id = doc_metadata.get('doc_id', '')
                for doc in complete_data:
                    if doc.get('doc_id') == doc_id:
                        evidence_sentences = self.extract_evidence_sentences([doc])
                        evidence_sentences = [self.clean_sentence(sentence) for sentence in evidence_sentences]
                        evidence_sentences_list.append({
                            'doc_id': doc_id,
                            'index': i + 1,
                            'evidence_sentences': evidence_sentences
                        })
                        break
            
            
            prompt = f"""
            Generate a **concise and coherent** summary of this medical study, focusing on **intervention**,**outcomes** and any **notable counts or observations** while ensuring brevity. The summary should **retain key terminology** to maximize accuracy but **eliminate redundancy and unnecessary details.**  

            The summary must be **clear, structured, and short yet informative**. It should **clearly convey** the **primary intervention, key outcomes, and significant conclusions** in a structured format, allowing a medical professional to quickly grasp the study's insights.  

            ### **IMPORTANT FORMATTING REQUIREMENTS:**  
            1. Summarize **faithfully from the study**, reusing medical terms when appropriate to preserve meaning.
            2. Include the **intervention**, **results or outcomes**, and any **important numeric findings or observations**.
            3. **Ensure fidelity to the extracted key sentences** by reusing relevant terms where appropriate.  
            4. **Preserve original wording** wherever feasible but **eliminate repetition and excess detail** to enhance clarity.  
            5. Write in a **continuous paragraph format** without bullet points, stars (*), or markdown formatting.  
            6. **Avoid section headers** like "Key Findings:" or "Clinical Relevance:".  
            7. Use **precise and professional language** suited for a medical audience.  
            8. **Keep the summary significantly shorter than {abstract}, ensuring conciseness without losing essential details.**  

            Study: {abstract}
            """

            response = ollama.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}]
            )   
            model_summary = response["message"]["content"].strip()
            model_summary = self.clean_summary_output(model_summary)

            relevance_prompt = f"""
                Rate this summary's relevance to '{query}' from 1-100.
                Output only the number, nothing else.
            ModelSummary: {model_summary}
            """
            relevance_response = ollama.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": relevance_prompt}]
            )
            
            relevance_text = relevance_response["message"]["content"].strip()
            digits = ''.join(filter(str.isdigit, relevance_text))
            relevance_score = int(digits) if digits else 0
            model_summary = model_summary.replace("Here is a single-line summary:", "")
            model_summary = model_summary.replace("The key finding is that", "")
            model_summary = model_summary.strip()

            
            summary_parts.append({
                'text': model_summary,
                'doc_id': doc_metadata.get('doc_id', ''),
                'index': i + 1,
                'relevance_score': relevance_score,
            })
        
            
        # Sort summaries by relevance score
        summary_parts.sort(key=lambda x: x['relevance_score'], reverse=True)
        final_summary_text = self.generate_final_summary(summary_parts, query)
        final_summary = [{
            'final_summary': final_summary_text,
            'doc_id': 'collective',  # A placeholder ID
            'index': 'All'  # Indicating this is a summary of all documents
        }]
        summaries = {
        'ExtractedSummary': extracted_summaries,
        'ModelSumm': summary_parts,
        'FinalSumm' : final_summary
        }
        return{
            'summaries': summaries,  
        }
    
    def generate_final_summary(self, summaries, query):
        """
        Generate a collective summary of all documents based on the LLM-generated summaries.
        
        Args:
            summaries: Dictionary containing the ModelSumm entries
            query: The search query
            
        Returns:
            str: A collective summary paragraph
        """
        # Get the model summaries, which are already sorted by relevance
        model_summaries = summaries
        
        if not model_summaries:
            return "No summaries available to generate a final summary."
        
        # Extract the text from each summary
        summary_texts = [summary['text'] for summary in model_summaries]
        
        # Create a context with the query and all summaries
        context = ""
        for i, summary in enumerate(summary_texts):
            context += f"Document {i+1}: {summary}\n\n"
        
        # Create a prompt for generating the final summary
        prompt = f"""
        Based on the following LLM-generated summaries of multiple medical research documents about "{query}", 
        Generate a **cohesive paragraph** that synthesizes the core findings from all these summaries.
        
        Your final summary should:
        1. **Integrate consistent and significant findings** across the documents.
        2. **Note any areas of consensus or contradiction** in the research.
        3. **Maintain scientific accuracy** and use professional medical terminology.
        4. Write in a **fluent, continuous paragraph** (no bullet points, numbering, or section headers).
        5. **Be concise yet comprehensive**
        6. Do **not** reference document numbers or include phrases like “Document 1 says...”.
        
        Here are the individual document summaries:
        
        {context}
        
        IMPORTANT: Your response should ONLY include the final summary paragraph. Do not include introductory 
        text, explanations, document numbers     or any other content besides the summary itself.
        """
        
        # Generate the final summary using Ollama
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        final_summary = response["message"]["content"].strip()
        
        # Clean the summary to ensure it's just a paragraph
        final_summary = self.clean_summary_output(final_summary)
        
        return final_summary
