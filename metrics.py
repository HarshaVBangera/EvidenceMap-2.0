import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import matplotlib
matplotlib.use('Agg')  # Set Matplotlib backend before importing pyplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import spacy
from rouge_score import rouge_scorer

class TextEvaluator:
    def __init__(self):
        self.rouge = Rouge()
        self.nlp = spacy.load("en_core_sci_lg")
        self.smoothie = SmoothingFunction().method4

    def calculate_rouge_scores(self, reference, candidate):
        """
        Calculate ROUGE scores between reference and candidate texts
        """
        try:
            scores = self.rouge.get_scores(candidate, reference)[0]
            return {
                'rouge-1': scores['rouge-1']['f'] * 100,
                'rouge-2': scores['rouge-2']['f'] * 100,
                'rouge-l': scores['rouge-l']['f'] * 100
            }
        except Exception as e:
            print(f"Error calculating ROUGE scores: {str(e)}")
            return {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0}

    def calculate_bleu_score(self, reference, candidate):
        """
        Calculate BLEU score between reference and candidate texts
        """
        try:
            reference_doc = self.nlp(reference.lower())
            candidate_doc = self.nlp(candidate.lower())
            
            reference_tokens = [token.text for token in reference_doc]
            candidate_tokens = [token.text for token in candidate_doc]
            
            score = sentence_bleu([reference_tokens], candidate_tokens,
                               weights=(0.25, 0.25, 0.25, 0.25),
                              smoothing_function=self.smoothie)
            return score * 100  # Convert to 0-100 scale
        except Exception as e:
            print(f"Error calculating BLEU score: {str(e)}")
            return 0

def main():
    file_path = 'newEval.xlsx'
      
    try:
        df = pd.read_excel(file_path)
        print(f"Successfully loaded data with {len(df)} rows")
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return

    evaluator = TextEvaluator()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    results = {
        'extracted_vs_manual': {'rouge1': [], 'rouge2': [], 'rougeL': [], 'bleu': []},
        'manual_vs_llm': {'rouge1': [], 'rouge2': [], 'rougeL': [], 'bleu': []}
    }

    for index, row in df.iterrows():
        doc_id = row.get('DocumentsID', f"Doc_{index}")
        extracted = str(row.get('Extracted Summary', ''))
        manual = str(row.get('Manual summary', ''))
        llm = str(row.get('LLM summary', ''))
        
        if not extracted or not manual or not llm:
            print(f"Skipping document {doc_id} due to missing summary")
            continue
            
        extracted_vs_manual = scorer.score(manual, extracted)
        manual_vs_llm = scorer.score(manual, llm)
        
        bleu_extracted_vs_manual = evaluator.calculate_bleu_score(manual, extracted)
        bleu_manual_vs_llm = evaluator.calculate_bleu_score(manual, llm)
        
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            results['extracted_vs_manual'][metric].append(extracted_vs_manual[metric].fmeasure)
            results['manual_vs_llm'][metric].append(manual_vs_llm[metric].fmeasure)
            
        results['extracted_vs_manual']['bleu'].append(bleu_extracted_vs_manual)
        results['manual_vs_llm']['bleu'].append(bleu_manual_vs_llm)

    avg_results = {comp: {metric: np.mean(scores) for metric, scores in metrics.items()} for comp, metrics in results.items()}
    
    print("\n===== EVALUATION RESULTS =====")
    for comparison, metrics in avg_results.items():
        print(f"\n{comparison}:")
        for metric, score in metrics.items():
            print(f"  {metric}: {score:.4f}")

    # Create separate dataframes for ROUGE and BLEU metrics
    rouge_data = []
    bleu_data = []
    
    for comp, metrics in avg_results.items():
        for metric, score in metrics.items():
            if metric == 'bleu':
                bleu_data.append({'Comparison': comp, 'Score': score})
            else:
                rouge_data.append({'Comparison': comp, 'Metric': metric, 'Score': score})
    
    rouge_df = pd.DataFrame(rouge_data)
    bleu_df = pd.DataFrame(bleu_data)
    
    # Create ROUGE plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Metric', y='Score', hue='Comparison', data=rouge_df)
    plt.title('ROUGE Evaluation Metrics')
    plt.ylim(0, 1)  # ROUGE scores are between 0 and 1
    plt.savefig('rouge_evaluation_results.png')
    plt.close()
    
    # Create BLEU plot
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Comparison', y='Score', data=bleu_df)
    plt.title('BLEU Evaluation Metrics')
    # Don't set ylim for BLEU to accommodate scores that might exceed 1
    plt.savefig('bleu_evaluation_results.png')
    plt.close()
    
    # Save detailed results
    detailed_results = []
    for i in range(len(results['extracted_vs_manual']['rouge1'])):
        row_data = {'Document': i+1}
        for comparison in results:
            for metric in results[comparison]:
                row_data[f"{comparison}_{metric}"] = results[comparison][metric][i]
        detailed_results.append(row_data)
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv('detailed_evaluation_results.csv', index=False)
    
    print("\nDetailed results saved to 'detailed_evaluation_results.csv'")
    print("ROUGE visualization saved to 'rouge_evaluation_results.png'")
    print("BLEU visualization saved to 'bleu_evaluation_results.png'")

if __name__ == "__main__":
    main()
