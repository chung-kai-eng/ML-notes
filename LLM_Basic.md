## In-context learning

- In-context learning (ICL) is a technique where task demonstrations are integrated into the prompt in a natural language format. This approach allows pre-trained LLMs to address new tasks without fine-tuning the model.

### Mathematical representation [(Link)](https://ai.stanford.edu/blog/understanding-incontext/)

- Pretraining distribution ($p$): assume that the pretraining data and LM are large enough that the LM fits the pretraining distribution exactly. Thus, use $p$ to denote both the pretraining distribution and the probability under the LM
- Prompt distribution: In-context learning prompts are lists of IID (independent and identically distributed) training examples concatenated together with one test input. Each example in the prompt is drawn as a sequence conditioned on the **same prompt concept**, which **describes the task to be learned**.

$$
p(output|prompt) = \int_{concept} p(output|concept, prompt)p(concept|prompt)d(concept)
$$

## Build a LLM application: [(Link)](https://docs.llamaindex.ai/en/stable/understanding/understanding.html)

1. Loading: source could be text files, pdfs, other websites, databases, or API
2. Indexing: create a data structure that enables you to search through the data. **Create vector embedding** to represent the meaning of the data
3. Storing: store the index and any associated metadata. This storage prevents the need for re-indexing in the future.
4. Querying: choose index strategy. (sub-queries, multi-step queries, hybrid approaches)
5. Evaluation: objective metrics on accuracy, reliability, and speed of your query responses.

## About different data source

- [LLaMA Hub data loaders for different data sources](https://llamahub.ai/?tab=loaders)
  - include database, pdf, json, ppt, markdown, some web, notion, csv

## About Model [(Link)](https://christophergs.com/blog/running-open-source-llms-in-python)

- Llama:
  - Adjust the context window: `max_tokens=512`: the max number of tokens
  - Temperature: control the creativity and randomness of the model responses
  - set up prompts:
    - Note that different models require different prompting formats
    - use JinJa Templates [(Link)](https://jinja.palletsprojects.com/en/3.1.x/)
      - template engine that allows for the dynamic generation of text-based formats such as HTML, XML, CSV, LaTeX, etc (also for prompt)
  - Formatting LLM output with **GBNF Grammars** [(Link)](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md)

## RAG

- an advanced form of prompt engineering
- Vector store is a key component of RAG
  - [Vector stores that LLaMA support](https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores.html)
    > pros:
  - no need to fine-tune again every time the new information is added
  - stricter access control and more visible
    - better interpretability and observability metrics as you can look at the context if a response doesn't seem right
      cons:
    - context window limitation

## Reference

- [Quick tutorial about LLaMA cpp](https://www.datacamp.com/tutorial/llama-cpp-tutorial)
- [In-context learning](<https://www.lakera.ai/blog/what-is-in-context-learning#:~:text=In%2Dcontext%20learning%20(ICL)%20is%20a%20technique%20where%20task,without%20fine%2Dtuning%20the%20model.>)
- [Understand why in-context learning work](https://ai.stanford.edu/blog/understanding-incontext/)
- [RAG & LLaMAIndex](https://christophergs.com/blog/ai-engineering-retrieval-augmented-generation-rag-llama-index#overview)
- [LangChain with LLaMAIndex](https://medium.com/@zekaouinoureddine/bring-your-own-data-to-llms-using-langchain-llamaindex-3ddbac8cc9eb)
- [RAG](https://christophergs.com/blog/ai-engineering-retrieval-augmented-generation-rag-llama-index)

## Complementary

- Jinja template

```python
# more tour in https://realpython.com/primer-on-jinja-templating/
template_str = """
Hello, {{ name }}!
{% if is_user_admin %}
Here are your admin options:
{% for option in admin_options %}
- {{ option }}
{% endfor %}
{% else %}
Here are your user options:
{% for option in user_options %}
- {{ option }}
{% endfor %}
{% endif %}
"""

# create the enviornment
env = Environment(loader=BaseLoader())
template = env.from_string(template_str)

prompt = template.render(
    name="Alice",
    is_user_admin=True,
    admin_options=["Manage Users", "Configure System", "View Logs"],
    user_options=["View Profile", "Change Password"]
)
print(prompt)
```





