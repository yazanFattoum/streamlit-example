import streamlit as st
#import openai

# Set up OpenAI API credentials
#openai.api_key = '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'

# Function to complete code
def complete_code(prompt):
    response = model.Completion.create(
        engine='text-davinci-003',  # Code-Davinci model
        prompt=prompt,
        max_tokens=100,  # Adjust as needed
        temperature=0.7,  # Adjust as needed
        n=5,  # Number of completions to generate
        stop=None   # Specify custom stop tokens if needed 
    )
    completions = [choice['text'].strip() for choice in response.choices]
    return completions

# Streamlit app
def main():
    st.title("Code Completion")

    # User input for code snippet
    prompt = st.text_area("Enter code snippet:", height=200)

    if st.button("Complete"):
        if prompt:
            completions = complete_code(prompt)
            st.subheader("Completions:")
            for i, completion in enumerate(completions, start=1):
                st.code(f"Completion {i}:\n{completion}", language="python")

# Run the Streamlit app
if __name__ == '__main__':
    main()
