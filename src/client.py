import os

from dotenv import load_dotenv
from openai import OpenAI

from type import QA

load_dotenv('../.env')

class OpenAIClient:

    def __init__(self) -> None:
        self.create_client()
        # ChatGPTの役割を読み出す
        self.load_situation('situation.txt')
        # ChatGPTへの指示を読み出す
        self.load_instruction('instruction.txt')

    def create_client(self) -> None:
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def load_situation(self, file: str) -> None:
        with open(file, 'r') as f:
            self.situation = f.read()

    def load_instruction(self, file: str) -> None:
        with open(file, 'r') as f:
            self.instruction = f.read()
    
    def get_response(self, question: str, qa_list: list[QA]) -> str:
        # プロンプトを作成する
        prompt = self.create_prompt(question, qa_list)
        # 送信したプロンプトを出力する
        self.output_prompt('prompt.txt', prompt)

        response = (
            self.openai_client
            # https://platform.openai.com/docs/api-reference/chat/create
            .chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": self.situation
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=1,
                max_tokens=512,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
        )

        return response.choices[0].message.content
    
    def create_prompt(self, question: str, qa_list: list[QA]) -> str:
        # 質問と回答例をプロンプトに整形する
        examples = list()
        for idx, qa in enumerate(qa_list):
            example = f'### 例{idx+1}\n質問: {qa.question}\n回答: {qa.answer}\n'
            examples.append(example)
        
        examples = '\n'.join(examples)

        # プロンプトを作成する
        prompt = (
            f'{self.instruction}\n'
            f'## 質問\n{question}\n\n'
            f'## 回答例\n{examples}\n'
            f'{self.instruction}'
        )
        
        return prompt
    
    def output_prompt(self, file: str, prompt: str) -> None:
        with open(file, 'w') as f:
            f.write(prompt)
