from transformers import pipeline

class ResultsSummarizer:
    def __init__(self):
        self.summarizer = pipeline("summarization", model="google/flan-t5-base")

    
    def generate_summary(self, collective_data, display_results):
        summary_parts = []
        
        for idx, result in enumerate(display_results, 1):
            # Get the main summary from abstract
            text = f"{result['title']} {result['abstract']}"
            summary = self.summarizer(
                text, 
                max_length=150,
                min_length=20, 
                do_sample=True, 
                temperature=0.7, 
                num_beams=4,
                early_stopping=True
            )
            
            # Extract study details from collective_data
            study_details = ""
            if isinstance(collective_data, dict) and 'data' in collective_data:
                study_data = collective_data['data'][idx-1]
                if 'study_results' in study_data:
                    interventions = [r['term'] for r in study_data['study_results'] if r['type'] == 'Intervention']
                    observations = [r['term'] for r in study_data['study_results'] if r['type'] == 'Observation']
                    outcomes = [r['term'] for r in study_data['study_results'] if r['type'] == 'Outcome']
                    
                    if interventions or observations or outcomes:
                        study_details = f" Results showed that {', '.join(interventions)} led to {', '.join(observations)} in {', '.join(outcomes)}."
            
            combined_summary = f"{summary[0]['summary_text']}{study_details}"
            
            summary_parts.append({
                'text': combined_summary.strip(),
                'doc_id': result['doc_id'],
                'index': idx
            })

        return {'summary_parts': summary_parts}