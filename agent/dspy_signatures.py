import dspy
import re


def extract_json_structure(text: str) -> str:
    """Extract JSON structure from text"""
    # Try to find JSON object
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    
    return text.strip()


# ==============================================================================
# 1. PARAMETER EXTRACTOR (Extracts only parameters, not full SQL)
# ==============================================================================

class ParameterExtractorSignature(dspy.Signature):
    """Extract structured parameters from question and document context.
    
    Only extract the parameters needed for SQL templates:
    - Dates (start_date, end_date) in YYYY-MM-DD format
    - Category name (exact match from question or docs)
    - Metric type (aov, revenue, margin, quantity)
    - Limit number (for TOP N queries)
    
    Do NOT generate SQL queries. Only extract the parameter values.
    """
    question: str = dspy.InputField()
    doc_context: str = dspy.InputField()
    
    start_date: str = dspy.OutputField(desc="Start date in YYYY-MM-DD format or empty string")
    end_date: str = dspy.OutputField(desc="End date in YYYY-MM-DD format or empty string")
    category: str = dspy.OutputField(desc="Product category name or empty string")
    metric: str = dspy.OutputField(desc="Metric type: aov, revenue, margin, quantity, or empty")
    limit: str = dspy.OutputField(desc="Limit number for TOP N queries or empty string")


class ParameterExtractor(dspy.Module):
    """Extract parameters for SQL templates using DSPy."""
    
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(ParameterExtractorSignature)
    
    def forward(self, question: str, doc_context: str):
        result = self.predictor(question=question, doc_context=doc_context)
        return result


# ==============================================================================
# 2. ANSWER SYNTHESIZER
# ==============================================================================

class SynthesizeAnswer(dspy.Signature):
    """Synthesize the final answer as JSON.
    
    CRITICAL INSTRUCTIONS:
    1. If sql_result is "No SQL data" and retrieved_chunks has no answer, 
       return {"answer": "I could not find data to answer this.", "source": "error"}.
    2. Do NOT invent information.
    3. Output valid JSON only with 'answer' and 'source' keys.
    4. The answer must match the format_hint exactly.
    """
    question: str = dspy.InputField()
    sql_result: str = dspy.InputField()
    retrieved_chunks: str = dspy.InputField()
    format_hint: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="JSON string with 'answer' and 'source' keys")


class AnswerSynthesizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(SynthesizeAnswer)
    
    def forward(self, question, sql_result, retrieved_chunks, format_hint):
        result = self.predictor(
            question=question,
            sql_result=sql_result,
            retrieved_chunks=retrieved_chunks,
            format_hint=format_hint
        )
        return result.answer
