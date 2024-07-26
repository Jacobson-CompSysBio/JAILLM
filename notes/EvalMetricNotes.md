# Evaluation Metrics
This markdown file contains information about various evaluation metrics for LLMs.
We broadly divide the metrics into two categories:  
    1. General Evaluation Metrics
    2. Biology-Specific Metrics
In each section, we include a description of the metrics, how they are calculated, and how to implement them in python.
HuggingFace has a [`metrics` package](https://huggingface.co/docs/datasets/en/about_metrics) which contains many of these.

Metrics generally fall under one or many of the following categories:

1. **Answer Relevancy:** Is the LLM able to address the given input in an informative and concise manner?

2. **Correctness:** Is the LLM factually correct, based on some ground truth?

3. **Hallucination:** Does output contain fake or made-up information?

4. **Contextual Relevancy:** Given a prior prompt or information, can the LLM recognize relevant context and provide a useful answer?

5. **Responsible Metrics:** Measure bias and toxicity; is the output harmful or offensive?

6. **Task-Specific Metrics:** Does the LLM output align with our biological goals?

## 1. General Evaluation Metrics
These are broad metrics generally used for evaluating LLMs. They are generally divided between **statistical** scorers and **model-based** scorers.

### Statistical Scorers
Statistical scorers involve the calculation of some metric between the LLM output and an expected ground truth. They are reliable, but due to the stochastic nature of LLMs, they may not capture reality as well as Model-Based Scorers

1. **BLEU (BiLingual Evaluation Understudy):** Evaluates the output of LLM application against annotated ground truths or expected outputs. Precision for each matching n-gram (n consective words) is calculated between actual and expected output, and a [geometric mean](https://en.wikipedia.org/wiki/Geometric_mean) is calculated. A brevity penalty can also be applied.
    * Can be applied with the `metrics` package

2. **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** Primarily used for evaluating text summaries; calculates recall by comparies overlap of n-grams between LLM actual and expected outputs. Determines the proportion of n-grams in the reference that are present in the LLM output
    * Can be applied with the `metrics` package
    * Could we use this to evaluate the quality of clade interpretations?

3. **METEOR (Metric for Evaluation of translation with Explicit Ordering):** More comprehensive; calculates scores by precision (n-gram matches) and recall (n-gram overlaps), adjusted for word order differences and synoyms. Final score is the [harmonic mean](https://en.wikipedia.org/wiki/Harmonic_mean) of precision and recall.
    * Can be applied with the `metrics` package


### Model-Based Scorers
Model-Based scorers rely on NLP models to evaluate outputs. They are comparatively more accurate but less reliable than statistical scorers due to their probabilistic nature.

1. **NLI (Natural Language Inference):** Uses a natural language model to assign a number, ranging from 0-1, the output's logical coherence, with 1 being more coherent and 0 being less.

2. **BLEURT (BLEU with Representations from Transformers):** Uses a pre-trained model like [BERT]() to score llm outputs based on some ground truth

NLI may struggle to understand long context, while BLEURT is only as good as the quality of its training data, and training both for specific purposes may be time-consuming.
