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

### Kendra 준비

AWS CDK를 이용하여 [Kendra 사용을 위한 준비](./kendra-preperation.md)와 같이 Kendra를 설치하고 사용할 준비를 합니다.

### Bedrock을 LangChain으로 연결하기

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

from langchain.llms.bedrock import Bedrock
llm = Bedrock(
    model_id=modelId, 
    client=boto3_bedrock, 
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    model_kwargs=parameters)
```

### 채팅이력을 저장하기 위한 메모리 설정

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

### 문서 등록

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

Query API와 다르게 qustion/answer와 FAG는 포함되지 않습니다. 

Kendra의 [FAQ((Frequently Asked Questions))를 이용](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/blob/main/kendra-faq.md)하면 RAG의 정확도를 개선할 수 있는데, Query API로만 결과를 얻을 수 있습니다. 또한, Kendra에서는 Retrieve API로 조회시 결과가 없을때에 Query API로 fallback을 best practice로 가이드하고 있습니다. 따라서, FAQ를 사용하고자 한다면, Retrive와 Query API를 모두 사용하여야 합니다.

[FAQ-Kendra](https://github.com/aws-samples/enterprise-search-with-amazon-kendra-workshop/blob/master/Part%202%20-%20Adding%20a%20FAQ.md)를 참조합니다. [kendra-faq-refresher](https://github.com/aws-samples/amazon-kendra-faq-refresher/tree/main)를 참조하여 FAQ를 Kendra 검색 결과로 활용할 수 있습니다.

여기서는 [Kendra FAQ](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/blob/main/kendra-faq.md)와 같이 Kendra의 Query API로 FAQ의 결과를 얻으면, 질문과 결과를 함께 excerpt로 활용합니다.


FAQ에 있는 "How many free clinics are in Spokane WA?"의 Answer는 13이고 아래와 같은 reference를 가지고 있습니다.

![noname](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/996174e6-765b-4d2c-a5ea-7cfeb838a609)

아래는 "How many clinics are in Spokane WA?"으로 Kenra에 Query 했을때의 결과입니다. 여기서 FAQ의 "free"가 있고 없음에 따라 결과는 매우 다름에도 불구하고 아래와 같이, Kendra의 응답은 "ScoreConfidence"를 "VERY_HIGH"로 응답하고 있습니다.

따라서, 응답의 Type이 "QUESTION_ANSWER"인 경우에는 발췌를 할때에 "AdditionalAttributes"의 "QuestionText"을 같이 사용하여야 합니다. 즉 "How many free clinics are in Spokane WA? 13"으로 사용합니다.

```python
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
                     "Highlights":[
                        {
                           "BeginOffset":4,
                           "EndOffset":8,
                           "TopAnswer":false,
                           "Type":"STANDARD"
                        },
                        {
                           "BeginOffset":14,
                           "EndOffset":21,
                           "TopAnswer":false,
                           "Type":"STANDARD"
                        },
                        {
                           "BeginOffset":29,
                           "EndOffset":36,
                           "TopAnswer":false,
                           "Type":"STANDARD"
                        },
                        {
                           "BeginOffset":37,
                           "EndOffset":39,
                           "TopAnswer":false,
                           "Type":"STANDARD"
                        }
                     ]
                  }
               }
            },
            {
               "Key":"AnswerText",
               "ValueType":"TEXT_WITH_HIGHLIGHTS_VALUE",
               "Value":{
                  "TextWithHighlightsValue":{
                     "Text":"13",
                     "Highlights":[
                        
                     ]
                  }
               }
            }
         ],
         "DocumentId":"c24c0fe9cbdfa412ac58d1b5fc07dfd4afd21cbd0f71df499f305296d985a8c9a91f1b2c-e28b-4d13-8b01-8a33be5fc126",
         "DocumentTitle":{
            "Text":""
         },
         "DocumentExcerpt":{
            "Text":"13",
            "Highlights":[
               {
                  "BeginOffset":0,
                  "EndOffset":2,
                  "TopAnswer":false,
                  "Type":"STANDARD"
               }
            ]
         },
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
         "FeedbackToken":"AYADeLqxw-UVD3GzCPho81xtW6IAXwABABVhd3MtY3J5cHRvLXB1YmxpYy1rZXkAREFrdE4vUzFJTWFmVGVjaFhUbHhlLzh4VXFvYXNablozR2RmeHVFb0JHN05ZYVVoNmZnMVRUMGdjQS9CTzdWWlNNQT09AAEAB2F3cy1rbXMAS2Fybjphd3M6a21zOnVzLXdlc3QtMjoxNDk0MDA5NDM5NTk6a2V5LzUyN2YwMjRhLTUyMDktNDI4NC1iOTYwLTJhMjYxMzQxNWNkNgC4AQIBAHhoFIrDBc0sA_W0qqJvieboGJWYBK_hEm739PftPtfwZwGkHCq8G_rQwpPcBduAGoQFAAAAfjB8BgkqhkiG9w0BBwagbzBtAgEAMGgGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMxTLvEAoHAoxFuktZAgEQgDsadJ5kv-qG2Clv0RRB7DSV8yK1Hv92wtCDYRDp0qswISUtpo6BlAvfaqAG3VD__jmy_wyv_iPuJpQn3wIAAAAADAAAEAAAAAAAAAAAAAAAAAAkeJmB-aXVfLQSLWk6xKfQ_____wAAAAEAAAAAAAAAAAAAAAEAAAF2YTptEGgTPKdRbRK31PKlnqy7fC4-exZfkWr1KhaYjExSOuoIxntlROmX8DaVLp7Iy-TaUrKg9-C0Iwj62FlMEDKbBxVdq7jI31uIZDP58Z17HUvl4acvRktyW_gaPMIDiVxo0QaAKvmP7qwq34-3Ti_ODBP3dTaufr7atTjsyBBJTQgJ4P3SSfMPniqdZOQTUnIb5PcbALwGVTT3FLxu9LxlpscfyoKvzGZSLAPDgmRWINEmPz9j9h-IzUATlJtqpydOnX3wAmUKx4GyBzISuhU7IxXK0BnYBAcwkl_ii04W5vEUi02cyLOBTU8iwYo-C0xUY7X2IccrzGHxF4o3YPW6mnmC_6YXm2nXPUi0OOPoX1sk5kuhx8Ra3n11DuKq6SAGH414hSxqFrPSfqBCIF5356z4tG_nJQ6KxKAi2LUwypAAme48DbZH_5KTsfFcm7X3MSr23XwNhq8jDbudlz-M3JF_6E314B7V99VASAxNxk-yiOUyjcMoECtcQ-cpgAbOf2MEAGcwZQIxAMtWD1KR4dhKMBJyJMOwoFsRgdVONBxgaz3z2TKJZfSUl3l-CFyENlB9oWq1f_dlOgIwXVTlmVXN5gyOZvvlqfvJybhNO2tyZ53ilC5_GsKl9Lu9cRnGjMX-tKqPyheSCVc3.6ca1e9c4-9ce1-41f7-a527-2b6a536ad8b4-e1b952fa-7d3b-4290-ac5e-9f12b99db3af"
      }
   ],
   "FacetResults":[
      
   ],
   "TotalNumberOfResults":1,
   "ResponseMetadata":{
      "RequestId":"f415076d-1384-47d4-8718-8245ca3c6d89",
      "HTTPStatusCode":200,
      "HTTPHeaders":{
         "x-amzn-requestid":"f415076d-1384-47d4-8718-8245ca3c6d89",
         "content-type":"application/x-amz-json-1.1",
         "content-length":"2642",
         "date":"Sat, 18 Nov 2023 03:24:27 GMT"
      },
      "RetryAttempts":0
   }
}
```

### Score 활용하기 

검색의 정확도(score)를 활용하여 검색의 범위를 조정하면 RAG의 정확도가 올라갑니다. 그런데, Retrieve는 2023년 11월(현재)까지 영어(en)에 대해서만 score를 제공하고 있습니다. 따라서, 한국어(ko)는 token수가 적은 Query API를 이용할때만 score를 활용할 수 있습니다.

[ScoreAttributes](https://docs.aws.amazon.com/kendra/latest/APIReference/API_ScoreAttributes.html)와 같이 "VERY_HIGH", "HIGH", "MEDIUM", "LOW", "NOT_AVAILABLE"로 결과의 신뢰도를 확인할 수 있습니다.


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


#### Query API

[Kendra의 query API](https://docs.aws.amazon.com/ko_kr/kendra/latest/APIReference/API_Query.html)를 이용하여, "DOCUMENT", "QUESTION_ANSWER", "ANSWER" Type의 결과를 얻을 수 있습니다.

- ANSWER: 관련 제안된 답변(Relevant suggested answers)으로 text나 table의 발취(excerpt)로서 강조 표시(highlight)를 지원합니다. 
- QUESTION_ANSWER: 관련된 FAQ(Matching FAQs) 또는 FAQ 파일에서 얻은 question-answer입니다.
- DOCUMENT: 관련된 문서(Relevant documents)로서 문서의 발취(excerpt)와 title을 포하한 결고로서 강조 표시(hightlight)를 지원합니다.

관련 파라메터는 아래와 같습니다.

- QueryResultTypeFilter로 type을 지정할 수 있습니다.
- PageSize: 관련된 문장을 몇개까지 가져올지 지정합니다.
- PageNumber: 기본값은 결과의 첫페이지입니다. 첫페이지 이후의 결과를 가져올때 지정합니다.

Retrieve API로 결과를 조회한 후에 Query API를 이용하여 FAQ를 조회합니다. 이때, [QueryResultTypeFilter](https://docs.aws.amazon.com/kendra/latest/APIReference/API_Query.html)을 "QUESTION_ANSWER"로 설정하면 FAQ의 결과만을 얻을 수 있습니다. 

```python
resp = kendra_client.query(
    IndexId = index_id,
    QueryText = query,
    PageSize = top_k / 2,
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





#### Langchain 활용하기

LangChain은 Retreive API로 검색하였을대에 결과가 없으면, Query로 한번 더 검색을 수행합니다.
Kendra에서 한국어 문서를 업로드하면 retriever의 Language 설정을 "ko"로 설정하여야 합니다.


### 정확도 개선 방안

- 한글문서의 언어설정을 "ko"로하여 Kendra에 등록합니다.
- LangChain의 Kendra Retriever로 질의시 language를 "ko"로 설정하여야 retriever api로 더 많은 token을 가지는 발췌문을 얻을 수 있습니다.
- Kendra의 Retrieve/Query API로 직접 조회하면 좀더 유연하게 RAG를 구현할 수 있습니다.
- FAQ문서가 있다면, Kendra에 등록하여 활용합니다. FAQ 사용시 Query API를 활용하여하므로, 결과를 얻는 속도를 개선하기 위해 동시에 Retrieve/Query API를 호출합니다.



## 직접 실습 해보기

### 사전 준비 사항

이 솔루션을 사용하기 위해서는 사전에 아래와 같은 준비가 되어야 합니다.

- [AWS Account 생성](https://repost.aws/ko/knowledge-center/create-and-activate-aws-account)에 따라 계정을 준비합니다.

 - [Characters in query text](https://us-west-2.console.aws.amazon.com/servicequotas/home/services/kendra/quotas/L-7107C1BC)에 접속하여 Kendra의 Query할수 있는 메시지의 사이즈를 3000으로 조정합니다.

### CDK를 이용한 인프라 설치
[인프라 설치](./deployment.md)에 따라 CDK로 인프라 설치를 진행합니다. [CDK 구현 코드](./cdk-chatbot-with-kendra/README.md)에서는 Typescript로 인프라를 정의하는 방법에 대해 상세히 설명하고 있습니다.


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

