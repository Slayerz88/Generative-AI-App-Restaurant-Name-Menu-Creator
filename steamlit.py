import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_google_genai import ChatGoogleGenerativeAI
import os

with open(r"D:\langchain\api_key.txt", "r") as f:
    api_key = f.read().strip()  # strip removes hidden \n or spaces

os.environ["GOOGLE_API_KEY"] = api_key


if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        google_api_key=api_key
    )

    # -----------------------------
    # Define function
    # -----------------------------
    def generate_restaurant_name_and_items(cuisine):
        # Chain 1: Restaurant Name
        prompt_template_name = PromptTemplate(
            input_variables=['cuisine'],
            template="I want to open a restaurant for {cuisine} food. Suggest a fancy name for this."
        )

        name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")

        # Chain 2: Menu Items
        prompt_template_items = PromptTemplate(
            input_variables=['restaurant_name'],
            template="""Suggest some menu items for {restaurant_name}. 
            Return it as a comma separated string"""
        )

        food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key="menu_items")

        chain = SequentialChain(
            chains=[name_chain, food_items_chain],
            input_variables=['cuisine'],
            output_variables=['restaurant_name', "menu_items"]
        )

        response = chain({'cuisine': cuisine})
        return response

    # -----------------------------
    # Streamlit UI
    # -----------------------------
    st.title("🍽️ AI Restaurant Name & Menu Generator")
    st.write("Enter a cuisine and get a fancy restaurant name with menu suggestions!")

    cuisine = st.text_input("Enter Cuisine Type (e.g., Italian, Indian, Japanese):")

    if st.button("Generate"):
        if cuisine.strip() == "":
            st.warning("Please enter a cuisine type.")
        else:
            with st.spinner("Generating..."):
                result = generate_restaurant_name_and_items(cuisine)
                st.subheader("✨ Restaurant Name:")
                st.success(result["restaurant_name"])
                st.subheader("🍴 Menu Items:")
                st.write(result["menu_items"])
else:
    st.warning("Please provide a valid Google API Key to continue.")
