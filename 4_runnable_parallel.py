from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel

llm = HuggingFaceEndpoint(
    repo_id= "moonshotai/Kimi-K2-Thinking",
    task="text-generation"
) 

model = ChatHuggingFace(llm=llm)

# We can aslo use another model too
llm2 = HuggingFaceEndpoint(
    repo_id='openai/gpt-oss-20b',
    task='text-generation'
)
model2 = ChatHuggingFace(llm=llm2)


prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a LinkedIn post about {topic}',
    input_variables=['topic']
)
parser = StrOutputParser()
parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkedIn': RunnableSequence(prompt2, model2, parser)
})

result = parallel_chain.invoke({
    'topic': 'AI',
})
# If you have different inputs to give then use like this: e.g.:
# result = parallel_chain.invoke({
#     'topic': 'AI',
#     'top': 'Health'
# })

# print(result)

print(result['tweet'])
print('\n', result['linkedIn'])