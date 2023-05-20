
import streamlit as st
import openai

# Set up OpenAI API
openai.api_key = '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'

# Define the prompt for code completion
prompt = """
You are a software developer working on a project. You need code completion suggestions to help you write code more efficiently.

Code:
def calculate_average(numbers):
    total = sum(numbers)
    average = total / len(numbers)
    return average

Please provide code completion suggestions for the following line:
average = 
"""

# Generate code completion suggestions
def generate_code_completion(prompt):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=100,
        n=5,  # Number of completions to generate
        stop=None,
        temperature=0.7
    )
    completions = [choice['text'].strip() for choice in response['choices']]
    return completions

# Streamlit app
def main():
    st.title("Code Completion with ChatGPT")

    # Display the prompt
    st.markdown("### Prompt:")
    st.code(prompt)

    # Generate code completions on button click
    if st.button("Get Code Completions"):
        completions = generate_code_completion(prompt)

        # Display code completion suggestions
        st.markdown("### Code Completions:")
        for i, completion in enumerate(completions):
            st.code(f"Suggestion {i+1}: {completion}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
