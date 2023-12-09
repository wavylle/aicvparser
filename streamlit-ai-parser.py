import openai
import streamlit as st
from pdfminer.high_level import extract_text
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import json
from openai import OpenAI

embeddings = OpenAIEmbeddings(openai_api_key="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")

prompt_template = """
        You are a helpful CV parser that returns name, email, phone number, skills, last company, qualification and years of experience of the candidate, based on the given text.
        Return a python dictionary with name, email, mobile_number, skills, last_company, qualification and experience as the keys. Skills and Qualification will be arrays.

        CV Text:
        {cv_text}"""

prompt = PromptTemplate(
    input_variables=["cv_text"],
    template=prompt_template,
)

# create chain using LLMChain
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key="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")

chain = LLMChain(llm=llm, prompt=prompt)

def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

def main():
    st.title("uKnowva CV Parser (Test Env)")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        text = extract_text_from_pdf(uploaded_file)

        screen_data = json.loads(chain.run(cv_text=text))
        print(screen_data)

        name = st.text_input("Name", value=screen_data["name"])
        email = st.text_input("Email", value=screen_data["email"])
        phone_number = st.text_input("Phone Number", value=screen_data["mobile_number"])

        # Create a multiselect widget for tags
        skills = st.multiselect("Skills", options=screen_data["skills"], default=screen_data["skills"])

        # Qualification
        qualification = st.multiselect("Skills", options=screen_data["qualification"], default=screen_data["qualification"])

        # Experience
        experience = st.text_input("Years of Experience", value=screen_data["experience"])

        last_company = st.text_input("Last Company", value=screen_data["last_company"])

        # Completion API
        client = OpenAI(
            # This is the default and can be omitted
            api_key="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",
        )

        response = client.chat.completions.create(messages=[
            {
                "role": "user",
                "content": f"""You are a helpful CV parser that returns name, email, phone number, skills, last company, qualification and years of experience of the candidate, based on the given text.
        Return a python dictionary with name, email, mobile_number, skills, last_company, qualification and experience as the keys. Skills and Qualification will be arrays.

        CV Text:
        {text}""",
            }
        ],
        model="gpt-3.5-turbo",
        )
        print(response)

if __name__ == "__main__":
    main()
