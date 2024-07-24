from client import OpenAIClient
from db import QADatabase

if __name__ == '__main__':
    qa_db = QADatabase()
    openai_client = OpenAIClient()
    
    while True:
        # ユーザーから質問を受け取る
        print('Enter a question. Type "quit" to exit.')
        question = input('>>> ')

        if question == 'quit':
            break
        
        # 類似した質問と回答を取得する
        similar_example = qa_db.get_similar_example([question])
        # 質問と共に回答例をChatGPTに送信する
        response = openai_client.get_response(question, similar_example)
        # ChatGPTからの回答を出力する
        print(response)
        print()
