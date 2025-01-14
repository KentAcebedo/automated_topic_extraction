import pathlib
import textwrap
import streamlit
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown

genai.configure(api_key="api")

text1 = "This document is about weight pig estim model imag"


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")

response = model.generate_content(["Make a short paragraph base on this", text1])

print(response)
