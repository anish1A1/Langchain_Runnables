from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough


llm = HuggingFaceEndpoint(
    repo_id='openai/gpt-oss-20b',
    task='text-generation'
)
model = ChatHuggingFace(llm=llm)


prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)
parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt, model, parser)

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)


parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explaination' : RunnableSequence(prompt2, model, parser)
})



final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
result = final_chain.invoke({'topic': 'GTA 5'})

# print(result)
print(result['joke'])
print('\n', result['explaination'])