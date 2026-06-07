from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional
import os

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    
    try:

        # Create evaluator LLM
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url="https://openai.vocareum.com/v1"
            )
        )

        # Create evaluator embeddings
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url="https://openai.vocareum.com/v1"
            )
        )

        # Define metrics
        faithfulness_metric = Faithfulness(
            llm=evaluator_llm
        )

        relevancy_metric = ResponseRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        )

        #bleu_metric = BleuScore()

        #rouge_metric = RougeScore()

        # Create sample
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts
        )

        # Evaluate
        results = {}

        results["faithfulness"] = faithfulness_metric.single_turn_score(
            sample
        )

        results["response_relevancy"] = relevancy_metric.single_turn_score(
            sample
        )

        #results["bleu"] = bleu_metric.single_turn_score(
        #    sample
        #)

        #results["rouge"] = rouge_metric.single_turn_score(
        #    sample
        #)

        return results

    except Exception as e:

        return {
            "error": str(e)
        }

    pass
