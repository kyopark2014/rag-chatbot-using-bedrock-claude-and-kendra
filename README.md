# Amazon Bedrock의 Claude와 Amazon Kendra를 이용하여 RAG가 적용된 Chatbot 만들기

기업의 데이터를 활용하여 질문과 답변을 수행하는 한국어 Chatbot을 RAG를 이용해 구현합니다. 이때 한국어 LLM으로는 Amazon Bedrock의 Claude 모델을 사용하고, 지식 저장소로 Amazon Kendra를 이용합니다. 

<img src="https://github.com/kyopark2014/rag-chatbot-using-bedrock-claude-and-kendra/assets/52392004/14ff613a-6a73-4125-b883-e295243ffa3e" width="800">


채팅 창에서 텍스트 입력(Prompt)를 통해 Kendra로 RAG를 활용하는 과정은 아래와 같습니다.
1) 사용자가 채팅창에서 질문(Question)을 입력합니다.
2) 이것은 Chat API를 이용하여 [lambda (chat)](./lambda-chat/index.js)에 전달됩니다.
3) lambda(chat)은 Kendra에 질문과 관련된 문장이 있는지 확인합니다.
4) Kendra로 부터 얻은 관련된 문장들로 prompt template를 생성하여 대용량 언어 모델(LLM) Endpoint로 질문을 전달합니다. 이후 답변을 받으면 사용자에게 결과를 전달합니다.
5) 결과는 DyanmoDB에 저장되어 이후 데이터 분석등의 목적을 위해 활용됩니다.

아래는 kendra를 이용한 메시지 동작을 설명하는 sequence diagram입니다. 

<img src="https://github.com/kyopark2014/rag-chatbot-using-bedrock-claude-and-kendra/assets/52392004/5ec32908-823b-47ea-baef-fbfba2ef240b" width="1000">

## 주요 구성

본 게시글의 Kendra를 이용한 검색 정확도를 높이기 위하여, FAQ와 ScoreAttributes를 활용합니다. 

Kendra는 자연어 검색을 통해 가장 유사한 문서의 발췌문을 제공하는데, 만약 관련된 단어나 유사한 의미가 없다고 하더라도 가장 관련된 문장을 선택하여 알려줍니다. 따라서, 때로는 관계가 없는 문장이 관련된 문장으로 선택되어 RAG의 정확도에 영향을 줄수 있습니다. 따라서, 검색의 관련도에 따른 score를 알수 있다면 RAG의 정확도를 향상 시킬 수 있습니다. Kendra의 Retrieve와 Query API는 [ScoreAttributes](https://docs.aws.amazon.com/kendra/latest/APIReference/API_ScoreAttributes.html)와 같이 "VERY_HIGH", "HIGH", "MEDIUM", "LOW", "NOT_AVAILABLE"로 검색 결과의 신뢰도를 확인할 수 있습니다. 하지만, Retrieve는 2023년 11월(현재)에 영어(en)에 대해서만 score를 제공하고 있습니다. 따라서, 본 게기글의 실습에서는 Query API의 ScoreAttribute를 활용하고 검색의 범위를 조정합니다.


### Kendra 준비

AWS CDK를 이용하여 [Kendra 사용을 위한 준비](./kendra-preperation.md)와 같이 Kendra를 설치하고 사용할 준비를 합니다.

### Bedrock의 Claude LLM을 LangChain으로 설정하기

아래와 같이 Langchain으로 Bedrock을 정의할때, Bedrock은 "us-east-1"으로 설정하고, Antrhopic의 Claude V2을 LLM으로 설정합니다.

```python
modelId = 'anthropic.claude-v2’
bedrock_region = "us-east-1" 

boto3_bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=bedrock_region,
    config=Config(
        retries = {
            'max_attempts': 30
        }            
    )
)

HUMAN_PROMPT = "\n\nHuman:"
AI_PROMPT = "\n\nAssistant:"
def get_parameter(modelId):
    if modelId == 'anthropic.claude-v1' or modelId == 'anthropic.claude-v2':
        return {
            "max_tokens_to_sample":8191, # 8k
            "temperature":0.1,
            "top_k":250,
            "top_p":0.9,
            "stop_sequences": [HUMAN_PROMPT]            
        }
parameters = get_parameter(modelId)

from langchain.llms.bedrock import Bedrock
llm = Bedrock(
    model_id=modelId, 
    client=boto3_bedrock, 
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    model_kwargs=parameters)
```

### 채팅이력을 저장하기 위한 메모리 준비 및 Dialog 저장

Lambda에 접속하는 사용자별로 채팅이력을 관리하기 위하여 [lambda-chatbot](./lambda-chat-ws/lambda_function.py)와 같이 map을 정의합니다. 클라이언트의 요청이 Lambda에 event로 전달되면, body에서 user ID를 추출하여 관련 채팅이력을 가진 메모리 맵이 없을 경우에는 [ConversationBufferWindowMemory](https://api.python.langchain.com/en/latest/memory/langchain.memory.buffer_window.ConversationBufferWindowMemory.html)을 이용해 정의합니다. 

```python
map_chain = dict()

jsonBody = json.loads(event.get("body"))
userId  = jsonBody['user_id']

if userId in map_chain:
    memory_chain = map_chain[userId]
else:
    memory_chain = ConversationBufferWindowMemory(memory_key="chat_history",output_key='answer',return_messages=True,k=5)
    map_chain[userId] = memory_chain
```

LLM을 통해 결과를 얻으면 아래와 같이 질문과 응답을 memory_chain에 새로운 dialog로 저장할 수 있습니다.

```python
memory_chain.chat_memory.add_user_message(text)
memory_chain.chat_memory.add_ai_message(msg)
```

### Kendra에 문서 등록하기

파일업로드후 링크를 제공할 수 있도록 파일이 저장된 S3의 파일명과 CloudFront의 도메인 주소를 이용하여 source_uri를 생성합니다. 이때 파일명에 공백등이 들어있을 수 있으므로 URL Encoding을 수행합니다. 또한, S3 Object의 파일 확장자를 추출해서 적절한 파일 타입으로 변환합니다. 

파일 속성으로 "_language_code"를 "ko"로 설정하였고 [batch_put_document()](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/kendra.html)을 이용하여 업로드를 수행합니다. 이때 S3를 이용해 업로드 할 수 있는 Document의 크기는 50MB이며, [문서포맷](https://docs.aws.amazon.com/kendra/latest/dg/index-document-types.html)와 같이 HTML, XML, TXT, CSV, JSON 뿐 아니라, Excel, Word, PowerPoint를 지원합니다.

```java
def store_document_for_kendra(path, s3_file_name, requestId):
    encoded_name = parse.quote(s3_file_name)
    source_uri = path + encoded_name    

    ext = (s3_file_name[s3_file_name.rfind('.')+1:len(s3_file_name)]).upper()
    if(ext == 'PPTX'):
        file_type = 'PPT'
    elif(ext == 'TXT'):
        file_type = 'PLAIN_TEXT'         
    elif(ext == 'XLS' or ext == 'XLSX'):
        file_type = 'MS_EXCEL'      
    elif(ext == 'DOC' or ext == 'DOCX'):
        file_type = 'MS_WORD'

    kendra_client = boto3.client(
        service_name='kendra', 
        region_name=kendra_region,
        config = Config(
            retries=dict(
                max_attempts=10
            )
        )
    )

    documents = [
        {
            "Id": requestId,
            "Title": s3_file_name,
            "S3Path": {
                "Bucket": s3_bucket,
                "Key": s3_prefix+'/'+s3_file_name
            },
            "Attributes": [
                {
                    "Key": '_source_uri',
                    'Value': {
                        'StringValue': source_uri
                    }
                },
                {
                    "Key": '_language_code',
                    'Value': {
                        'StringValue': "ko"
                    }
                },
            ],
            "ContentType": file_type
        }
    ]

    kendra_client.batch_put_document(
        IndexId = kendraIndex,
        RoleArn = roleArn,
        Documents = documents       
    )
```


### FAQ 활용하기

자주 사용하는 질문과 답변을 Kendra의 [FAQ((Frequently Asked Questions)](https://docs.aws.amazon.com/kendra/latest/dg/in-creating-faq.html#using-faq-file)로 등록하여 놓으면, RAG의 정확도를 개선할 수 있습니다. [FAQ-Kendra](https://github.com/aws-samples/enterprise-search-with-amazon-kendra-workshop/blob/master/Part%202%20-%20Adding%20a%20FAQ.md)와 같이 Kendra Console에서 FAQ를 등록할 수 있습니다. 아래의 [FAQ 예제](./contents/faq/demo.csv)를 등록후에 "How many free clinics are in Spokane WA?"를 질문하면 답변은 13이고, 참고 자료에 대한 uri를 확인할 수 있습니다.

![noname](https://github.com/kyopark2014/rag-chatbot-using-bedrock-claude-and-kendra/assets/52392004/e271ba1e-3b7c-4f44-bf9f-b07bdaf89a34)

Kendra의 FAQ는 Query API를 이용하고 검색하고, 아래와 같이 질문('QuestionText'), 답변('AnswerText'), URI('_source_uri')에 대한 정보뿐 아니라, 'ScoreConfidence'로 'VERY_HIGH'을 얻을 수 있습니다. [ScoreAttributes](https://docs.aws.amazon.com/kendra/latest/APIReference/API_ScoreAttributes.html)는 "VERY_HIGH", "HIGH", "MEDIUM", "LOW", "NOT_AVAILABLE"로 결과의 신뢰도를 제공합니다. 따라서, 'ScoreConfidence'의 범위를 제한하면 좀더 신뢰할만한 관련문서를 얻을 수 있습니다.

```java
{
   "QueryId":"6ca1e9c4-9ce1-41f7-a527-2b6a536ad8b4",
   "ResultItems":[
      {
         "Id":"6ca1e9c4-9ce1-41f7-a527-2b6a536ad8b4-e1b952fa-7d3b-4290-ac5e-9f12b99db3af",
         "Type":"QUESTION_ANSWER",
         "Format":"TEXT",
         "AdditionalAttributes":[
            {
               "Key":"QuestionText",
               "ValueType":"TEXT_WITH_HIGHLIGHTS_VALUE",
               "Value":{
                  "TextWithHighlightsValue":{
                     "Text":"How many free clinics are in Spokane WA?",
                  }
               }
            },
            {
               "Key":"AnswerText",
               "ValueType":"TEXT_WITH_HIGHLIGHTS_VALUE",
               "Value":{
                  "TextWithHighlightsValue":{
                     "Text":"13",
                  }
               }
            }
         ],
         "DocumentId":"c24c0fe9cbdfa412ac58d1b5fc07dfd4afd21cbd0f71df499f305296d985a8c9a91f1b2c-e28b-4d13-8b01-8a33be5fc126",
         "DocumentURI":"https://www.freeclinics.com/",
         "DocumentAttributes":[
            {
               "Key":"_source_uri",
               "Value":{
                  "StringValue":"https://www.freeclinics.com/"
               }
            }
         ],
         "ScoreAttributes":{
            "ScoreConfidence":"VERY_HIGH"
         },
         "FeedbackToken":"AYADeLqxw- 중략 - sKl9Lud8b4-e1b952fa-7d3b-4290-ac5e-9f12b99db3af"
      }
   ],
   "TotalNumberOfResults":1,
}
```

Kendra는 FAQ들 중에 가장 가까운 답을 주는데, "How many clinics are in Spokane WA?"와 같이 "free"를 빼고 입력하면 전혀 다른 결과를 주어야 하나, Kendra는 "ScoreConfidence"를 "VERY_HIGH"로 "13" 응답합니다. 따라서, Kendra의 FAQ 답변을 그대로 사용하지 말고, 결과에서 질문과 답변으로 "How many free clinics are in Spokane WA? 13"와 같은 문장을 만들어서 RAG에서 관련 문서(relevant doc)로 활용합니다.


### Kendra에서 문서 조회하기

Kendra에서 검색할때에 사용하는 API에는 [Retrieve](https://docs.aws.amazon.com/kendra/latest/APIReference/API_Retrieve.html)와 [Query](https://docs.aws.amazon.com/ko_kr/kendra/latest/APIReference/API_Query.html)가 있습니다. Retrieve API는 Query API 보다 더 큰 수의 token 숫자를 가지는 발췌를 제공하므로 일반적으로 더 나은 결과를 얻습니다. LangChain의 [AmazonKendraRetriever](https://api.python.langchain.com/en/latest/_modules/langchain/retrievers/kendra.html#AmazonKendraRetriever)은 먼저 retrieve API를 사용한 후에 결과가 없으면 query API로 fallback을 수행합니다. 

본 게시글에서는 Kendra의 검색정확도를 높이기 위하여, [Kendra의 FAQ](https://docs.aws.amazon.com/kendra/latest/dg/in-creating-faq.html#using-faq-file)와 [ScoreAttributes](https://docs.aws.amazon.com/kendra/latest/APIReference/API_ScoreAttributes.html)를 활용하기 위하여 LangChain의 [RetrievalQA](https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html?highlight=retrievalqa#), [ConversationalRetrievalChain](https://api.python.langchain.com/en/latest/chains/langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain.html#)을 이용하지 않고, [Prompt](https://api.python.langchain.com/en/latest/api_reference.html?highlight=prompt#module-langchain.prompts)를 이용해 동일한 동작을 구현하였습니다. 

[Retrieve](https://docs.aws.amazon.com/kendra/latest/APIReference/API_Retrieve.html)는 Default Quota 기준으로 하나의 발췌문(passges)는 200개의 token으로 구성될 수 있고, 최대 100개(PageSize)까지 이런 발췌문을 얻을 수 있습니다. 200 개의 token으로 구성된 발췌문(passage)과 최대 100개의 의미론적으로 관련된 발췌문을 검색할 수 있습니다. [Retrieve API는 2023년 11월 현재에 영어(en)만 Confidence score를 제공](https://docs.aws.amazon.com/kendra/latest/APIReference/API_Retrieve.html)합니다. 또한, Kendra 검색 성능을 개선하기 위해 사용하는 [feedback](https://docs.aws.amazon.com/kendra/latest/dg/submitting-feedback.html)도 지원하지 않습니다.



  
파일을 Kendra에 넣을때에 "_language_code"을 "ko"로 설정하였으므로, retrieve API를 이용하여 관련 문서를 검색할 때에도 동일하게 설정합니다. [Document Attribute](https://docs.aws.amazon.com/kendra/latest/dg/hiw-document-attributes.html)에 따라 "_source_uri", "_excerpt_page_number" 등을 설정합니다. 

```python
resp = kendra_client.retrieve(
    IndexId = index_id,
    QueryText = query,
    PageSize = top_k,
    AttributeFilter = {
        "EqualsTo": {
            "Key": "_language_code",
            "Value": {
                "StringValue": "ko"
            }
        },
    },
)
query_id = resp["QueryId"]

if len(resp["ResultItems"]) >= 1:
    retrieve_docs = []
    for query_result in resp["ResultItems"]:
        confidence = query_result["ScoreAttributes"]['ScoreConfidence']
    
    if confidence == 'VERY_HIGH' or confidence == 'HIGH':
        retrieve_docs.append(extract_relevant_doc_for_kendra(query_id = query_id, apiType = "retrieve", query_result = query_result))

def extract_relevant_doc_for_kendra(query_id, apiType, query_result):
    rag_type = "kendra"
    if(apiType == 'retrieve'): # retrieve API
        excerpt = query_result["Content"]
        confidence = query_result["ScoreAttributes"]['ScoreConfidence']
        document_id = query_result["DocumentId"] 
        document_title = query_result["DocumentTitle"]
        
        document_uri = ""
        document_attributes = query_result["DocumentAttributes"]
        for attribute in document_attributes:
            if attribute["Key"] == "_source_uri":
                document_uri = str(attribute["Value"]["StringValue"])        
        if document_uri=="":  
            document_uri = query_result["DocumentURI"]

        doc_info = {
            "rag_type": rag_type,
            "api_type": apiType,
            "confidence": confidence,
            "metadata": {
                "document_id": document_id,
                "source": document_uri,
                "title": document_title,
                "excerpt": excerpt,
            },
        }

return doc_info
```


[Kendra의 query API](https://docs.aws.amazon.com/ko_kr/kendra/latest/APIReference/API_Query.html)를 이용하여, 'QueryResultTypeFilter'를 "QUESTION_ANSWER"로 지정하면, FAQ의 결과만을 얻을 수 있습니다. 컨텐츠를 등록할때 "_language_code"을 "ko"로 지정하였으므로, 동일하게 설정합니다. PageSize는 몇개의 문장을 가져올것인지를 지정하는 것으로서 Retrieve와 Query 결과를 모두 relevant document로 사용하기 위해 전체의 반으로 설정하였습니다. 여기서는 FAQ중에 관련도가 높은것만 활용하기 위하여, ScoreConfidence가 "VERY_HIGH"인 문서들만 relevant docs로 활용하고 있습니다. 

```python
resp = kendra_client.query(
    IndexId = index_id,
    QueryText = query,
    PageSize = 4,
    QueryResultTypeFilter = "QUESTION_ANSWER",  
    AttributeFilter = {
        "EqualsTo": {
            "Key": "_language_code",
            "Value": {
                "StringValue": "ko"
            }
        },
    },
)
print('query resp:', json.dumps(resp))
query_id = resp["QueryId"]

if len(resp["ResultItems"]) >= 1:
    for query_result in resp["ResultItems"]:
        confidence = query_result["ScoreAttributes"]['ScoreConfidence']
    if confidence == 'VERY_HIGH':
        relevant_docs.append(extract_relevant_doc_for_kendra(query_id=query_id, apiType="query", query_result=query_result))    
    if len(relevant_docs) >= top_k:
        break
```

### 채팅이력을 이용하여 새로운 질문 생성하기

채팅화면에서 대화에서 Q&A를 수행하려면, 이전 채팅 이력과 현재의 질문을 이용하여 새로운 질문을 생성하여야 합니다. 여기서는 질문이 한글/영어 인지를 확인하여 다른 Prompt를 이용하여 새로운 질문(revised_question)을 생성합니다. 

```python
def get_revised_question(connectionId, requestId, query):        
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')  # check korean
    word_kor = pattern_hangul.search(str(query))

    if word_kor and word_kor != 'None':
        condense_template = """{chat_history}
        Human: 이전 대화와 다음의 <question>을 이용하여, 새로운 질문을 생성하여 질문만 전달합니다.

        <question>            
        {question}
        </question>
            
        Assistant: 새로운 질문:"""
    else: 
        condense_template = """{chat_history}    
        Answer only with the new question.

        Human: How would you ask the question considering the previous conversation: {question}

        Assistant: Standalone question:"""

    condense_prompt_claude = PromptTemplate.from_template(condense_template)        
    condense_prompt_chain = LLMChain(llm=llm, prompt=condense_prompt_claude)

    chat_history = extract_chat_history_from_memory()
    revised_question = condense_prompt_chain.run({"chat_history": chat_history, "question": query})
    
    return revised_question
```    

### RAG를 이용한 결과 확인하기

Kendra에 top_k개의 관련된 문서를 요청하여 받은 후에 아래와 같이 발취문(excerpt)를 추출하여 한개의 relevant_context를 생성합니다. 이후 아래와 같이 RAG용으로 만든 Prompt를 생성합니다. 

```python
relevant_docs = retrieve_from_Kendra(query=revised_question, top_k=top_k)

relevant_context = ""
for document in relevant_docs:
    relevant_context = relevant_context + document['metadata']['excerpt'] + "\n\n"

PROMPT = get_prompt_template(revised_question, convType)
def get_prompt_template(query, convType):
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    word_kor = pattern_hangul.search(str(query))

    if word_kor and word_kor != 'None':
        prompt_template = """\n\nHuman: 다음의 <context>를 참조하여 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다.
        
        <context>
        {context}
        </context>

        <question>
        {question}
        </question>

        Assistant:"""
                
    else:  # English
        prompt_template = """\n\nHuman: Here is pieces of context, contained in <context> tags. Provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            
        <context>
        {context}
        </context>
                        
        <question>
        {question}
        </question>

        Assistant:"""
return PromptTemplate.from_template(prompt_template)
```

Prompt를 이용하여 관련된 문서를 context로 제공하고 새로운 질문(revised_question)을 전달한 후에 응답을 확인합니다. 관련된 문서(relevant_docs)에서 "title", "_excerpt_page_number", "source"를 추출하여 reference로 추가합니다. 이때 FAQ의 경우는 source uri를 제공할 수 없으므로 아래와 같이 alert으로 FAQ의 quetion/answer 정보를 화면에 보여줍니다. 

```python
try: 
    stream = llm(PROMPT.format(context=relevant_context, question=revised_question))
    msg = readStreamMsg(connectionId, requestId, stream)
except Exception:
    raise Exception ("Not able to request to LLM")    

if len(relevant_docs)>=1 and enableReference=='true':
    msg = msg+get_reference(relevant_docs)

def get_reference(docs):
    reference = "\n\nFrom\n"
    for i, doc in enumerate(docs):
        if doc['api_type'] == 'retrieve': # Retrieve. socre of confidence is only avaialbe for English
            uri = doc['metadata']['source']
            name = doc['metadata']['title']
            reference = reference + f"{i+1}. <a href={uri} target=_blank>{name} </a>\n"
        else: # Query
            confidence = doc['confidence']
            if ("type" in doc['metadata']) and (doc['metadata']['type'] == "QUESTION_ANSWER"):
                excerpt = str(doc['metadata']['excerpt']).replace('"'," ") 
                reference = reference + f"{i+1}. <a href=\"#\" onClick=\"alert(`{excerpt}`)\">FAQ ({confidence})</a>\n"
            else:
                uri = ""
                if "title" in doc['metadata']:
                    name = doc['metadata']['title']
                    if name: 
                        uri = path+parse.quote(name)

                page = ""
                if "document_attributes" in doc['metadata']:
                    if "_excerpt_page_number" in doc['metadata']['document_attributes']:
                        page = doc['metadata']['document_attributes']['_excerpt_page_number']
                                        
                if page: 
                    reference = reference + f"{i+1}. {page}page in <a href={uri} target=_blank>{name} ({confidence})</a>\n"
                elif uri:
                    reference = reference + f"{i+1}. <a href={uri} target=_blank>{name} ({confidence})</a>\n"        
    return reference
```

## 직접 실습 해보기

### 사전 준비 사항

이 솔루션을 사용하기 위해서는 사전에 아래와 같은 준비가 되어야 합니다.

- [AWS Account 생성](https://repost.aws/ko/knowledge-center/create-and-activate-aws-account)에 따라 계정을 준비합니다.

 - [Characters in query text](https://us-west-2.console.aws.amazon.com/servicequotas/home/services/kendra/quotas/L-7107C1BC)에 접속하여 Kendra의 Query할수 있는 메시지의 사이즈를 3000으로 조정합니다.

### CDK를 이용한 인프라 설치
[인프라 설치](./deployment.md)에 따라 CDK로 인프라 설치를 진행합니다. [CDK 구현 코드](./cdk-chatbot-with-kendra/README.md)에서는 Typescript로 인프라를 정의하는 방법에 대해 상세히 설명하고 있습니다.

설치가 완료되면 아래와 같이 Output을 확인할 수 있습니다. 

![image](https://github.com/kyopark2014/rag-chatbot-using-bedrock-claude-and-kendra/assets/52392004/df0c08c2-0740-4558-a1d0-9f228f9726bc)

FAQ를 생성하기 위하여 아래에서 FAQUpdateforkoreanchatbot의 명령어를 복사해서 터미널에 붙여 넣기 합니다.

![image](https://github.com/kyopark2014/rag-chatbot-using-bedrock-claude-and-kendra/assets/52392004/71e96a30-1ecb-4f06-9e8e-d459f204fb21)

Kendra console의 [FAQs]에 접속하면 아래와 같이 "FAQ_fsi"로 FAQ가 등록된것을 확인할 수 있습니다.

![image](https://github.com/kyopark2014/rag-chatbot-using-bedrock-claude-and-kendra/assets/52392004/f1422fef-768e-4a40-874b-26686ef5699c)

이후 Output의 WebUrlforkoreanchatbot에 있는 URL을 복사하여 웹브라우저로 접속합니다.

## 실행결과



#### Q&A Chatbot 시험 결과

[fsi_faq_ko.csv](https://github.com/kyopark2014/question-answering-chatbot-with-vector-store/blob/main/fsi_faq_ko.csv)을 다운로드 하고, 채팅창의 파일 아이콘을 선택하여 업로드합니다.

![image](https://github.com/kyopark2014/question-answering-chatbot-with-kendra/assets/52392004/b35681ea-0f94-49cc-96ca-64b27df0fad6)


채팅창에 "이체를 할수 없다고 나옵니다. 어떻게 해야 하나요?” 라고 입력하고 결과를 확인합니다.

![image](https://github.com/kyopark2014/question-answering-chatbot-with-kendra/assets/52392004/f51a0cbf-a337-44ed-9a14-dd46e7fa7a6c)


채팅창에 "간편조회 서비스를 영문으로 사용할 수 있나요?” 라고 입력합니다. "영문뱅킹에서는 간편조회서비스 이용불가"하므로 좀더 자세한 설명을 얻었습니다.


![image](https://github.com/kyopark2014/question-answering-chatbot-with-kendra/assets/52392004/28b5c489-fa35-4896-801c-4609ebb68266)


채팅창에 "공동인증서 창구발급 서비스는 무엇인가요?"라고 입력하고 결과를 확인합니다.

![image](https://github.com/kyopark2014/question-answering-chatbot-with-kendra/assets/52392004/a0024c28-a0a4-4f18-b459-a9737c95db77)



#### Chat Hisity 활용의 예

chat history에 "안녕. 나는 서울에 살고 있어. "와 같이 입력하여 현재 서울에 살고 있음을 기록으로 남깁니다.

![image](https://github.com/kyopark2014/question-answering-chatbot-with-kendra/assets/52392004/074fe1cc-71e0-4a6d-baff-a2a8c7577f5c)

"내가 사는 도시에 대해 설명해줘."로 질문을 하면 chat history에서 서울에 대한 정보를 가져와서 아래와 같이 답변하게 됩니다. 

![image](https://github.com/kyopark2014/question-answering-chatbot-with-kendra/assets/52392004/fe1c7be4-319c-445f-ae9c-46f61914c48a)

이때의 로그를 보면 아래와 같이 입력한 질문("내가 사는 도시에 대해 설명해줘.")이 아래와 같이 "서울에 대해 설명해 주세요."와 같이 새로운 질문으로 변환된것을 알 수 있습니다.

```text
generated_prompt:   서울에 대해 설명해 주세요.
```


## Reference 

[Kendra - LangChain](https://python.langchain.com/docs/integrations/retrievers/amazon_kendra_retriever)

[kendra_chat_anthropic.py](https://github.com/aws-samples/amazon-kendra-langchain-extensions/blob/main/kendra_retriever_samples/kendra_chat_anthropic.py)

[IAM access roles for Amazon Kendra](https://docs.aws.amazon.com/kendra/latest/dg/iam-roles.html)

[Adding documents with the BatchPutDocument API](https://docs.aws.amazon.com/kendra/latest/dg/in-adding-binary-doc.html)

[class CfnIndex (construct)](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-cdk-lib.aws_kendra.CfnIndex.html)

[boto3 - batch_put_document](https://boto3.amazonaws.com/v1/documentation/api/1.26.99/reference/services/kendra/client/batch_put_document.html)

